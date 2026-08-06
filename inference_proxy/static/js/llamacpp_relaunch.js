// Poll-safe state and validation for managed llama.cpp runtime relaunches.

var LLAMACPP_RELAUNCH_LIMITS = Object.freeze({
  contextAlignment: 256,
  maxSlots: 256,
  maxAggregateContext: 4294967040,
});

function createLlamaCppRelaunchController() {
  var fields = {
    sizing: "auto",
    fit_target_mib: "",
    context_per_slot: "",
    slots: "",
    cache_type: "f16",
  };
  var sourceRuntime = null;
  var latestRuntime = null;
  var sourceSignature = null;
  var dirty = false;
  var stale = false;
  var submissionState = "idle";
  var previousTaskStartedAt = null;
  var trackedTaskStartedAt = null;

  var RELAUNCH_STEPS = new Set([
    "relaunch_validating",
    "draining",
    "stopping_llamacpp",
    "starting_llamacpp",
    "health_poll",
    "registering",
    "rolling_back",
  ]);
  var TERMINAL_STEPS = new Set(["complete", "failed"]);

  function runtimeSignature(runtime) {
    if (!runtime) return null;
    return JSON.stringify({
      requested: runtime.requested,
      effective: runtime.effective,
      gpus: runtime.gpus,
      observed_at: runtime.observed_at,
    });
  }

  function clonePolicy(policy) {
    var result = {
      sizing: policy.sizing,
      fit_target_mib: policy.fit_target_mib,
    };
    if (policy.sizing === "custom") {
      result.context_per_slot = policy.context_per_slot;
      result.slots = policy.slots;
      result.cache_type = policy.cache_type;
    }
    return result;
  }

  function policiesEqual(left, right) {
    if (!left || !right) return false;
    if (left.sizing !== right.sizing) return false;
    if (left.fit_target_mib !== right.fit_target_mib) return false;
    if (left.sizing === "auto") return true;
    return left.context_per_slot === right.context_per_slot &&
      left.slots === right.slots &&
      left.cache_type === right.cache_type;
  }

  function seed(runtime) {
    var requested = runtime.requested;
    var effective = runtime.effective;
    fields.sizing = requested.sizing;
    fields.fit_target_mib = String(requested.fit_target_mib);
    fields.context_per_slot = String(
      requested.context_per_slot == null
        ? effective.context_per_slot
        : requested.context_per_slot
    );
    fields.slots = String(
      requested.slots == null ? effective.slots : requested.slots
    );
    fields.cache_type = requested.cache_type || effective.cache_type_k;
    sourceRuntime = runtime;
    latestRuntime = runtime;
    sourceSignature = runtimeSignature(runtime);
    dirty = false;
    stale = false;
  }

  function reconcileRuntime(runtime, acceptTrackedResult) {
    latestRuntime = runtime;
    var nextSignature = runtimeSignature(runtime);
    if (sourceSignature === null) {
      seed(runtime);
      return "seeded";
    }
    if (nextSignature === sourceSignature) return "unchanged";
    if (acceptTrackedResult || !dirty) {
      seed(runtime);
      return "seeded";
    }
    stale = true;
    return "stale";
  }

  function integerValue(value) {
    var text = String(value).trim();
    if (!/^[0-9]+$/.test(text)) return null;
    var parsed = Number(text);
    return Number.isSafeInteger(parsed) ? parsed : null;
  }

  function validate() {
    var errors = {};
    var fitTarget = integerValue(fields.fit_target_mib);
    var runtime = latestRuntime || sourceRuntime;
    var minimumTotal = runtime
      ? Math.min.apply(null, runtime.gpus.map(function (gpu) {
        return gpu.total_mib;
      }))
      : null;

    if (fields.sizing !== "auto" && fields.sizing !== "custom") {
      errors.sizing = "Choose Automatic or Custom sizing.";
    }
    if (fitTarget === null || fitTarget < 1) {
      errors.fit_target_mib = "Free VRAM target must be a positive integer MiB value.";
    } else if (minimumTotal !== null && fitTarget >= minimumTotal) {
      errors.fit_target_mib = "Free VRAM target must be smaller than every GPU's total VRAM.";
    }

    if (Object.keys(errors).length || fields.sizing === "auto") {
      return {
        valid: Object.keys(errors).length === 0,
        errors: errors,
        body: Object.keys(errors).length ? null : {
          sizing: "auto",
          fit_target_mib: fitTarget,
        },
      };
    }

    var context = integerValue(fields.context_per_slot);
    var slots = integerValue(fields.slots);
    var trainContext = runtime ? runtime.effective.train_context : null;
    if (context === null || context < LLAMACPP_RELAUNCH_LIMITS.contextAlignment) {
      errors.context_per_slot = "Context per slot must be at least 256 tokens.";
    } else if (context % LLAMACPP_RELAUNCH_LIMITS.contextAlignment !== 0) {
      errors.context_per_slot = "Context per slot must use 256-token increments.";
    } else if (trainContext !== null && context > trainContext) {
      errors.context_per_slot = "Context per slot cannot exceed the model training context.";
    }
    if (slots === null || slots < 1 || slots > LLAMACPP_RELAUNCH_LIMITS.maxSlots) {
      errors.slots = "Parallel slots must be an integer from 1 through 256.";
    }
    if (fields.cache_type !== "f16" && fields.cache_type !== "q8_0") {
      errors.cache_type = "KV cache must be F16 or Q8_0.";
    }
    if (context !== null && slots !== null &&
        context * slots > LLAMACPP_RELAUNCH_LIMITS.maxAggregateContext) {
      errors.context_per_slot = "Context per slot multiplied by slots exceeds llama.cpp's aggregate context limit.";
    }

    return {
      valid: Object.keys(errors).length === 0,
      errors: errors,
      body: Object.keys(errors).length ? null : {
        sizing: "custom",
        fit_target_mib: fitTarget,
        context_per_slot: context,
        slots: slots,
        cache_type: fields.cache_type,
      },
    };
  }

  function recalculateDirty() {
    if (!sourceRuntime) {
      dirty = false;
      return;
    }
    var result = validate();
    dirty = !result.valid || !policiesEqual(
      result.body,
      clonePolicy(sourceRuntime.requested)
    );
  }

  function setField(name, value) {
    if (!Object.prototype.hasOwnProperty.call(fields, name)) return;
    fields[name] = String(value);
    recalculateDirty();
  }

  function reset() {
    if (latestRuntime) seed(latestRuntime);
  }

  function preview() {
    var context = integerValue(fields.context_per_slot);
    var slots = integerValue(fields.slots);
    var aggregate = context === null || slots === null ? null : context * slots;
    return {
      aggregate_context: aggregate,
      exceeds_aggregate_limit: aggregate !== null &&
        aggregate > LLAMACPP_RELAUNCH_LIMITS.maxAggregateContext,
    };
  }

  function taskStartedAt(task) {
    return task && typeof task.started_at === "string" ? task.started_at : null;
  }

  function isBusy() {
    return submissionState === "posting" ||
      submissionState === "waiting" ||
      submissionState === "active" ||
      submissionState === "unknown";
  }

  function beginSubmission(task) {
    if (isBusy()) return false;
    previousTaskStartedAt = taskStartedAt(task);
    trackedTaskStartedAt = null;
    submissionState = "posting";
    return true;
  }

  function markAccepted() {
    if (submissionState === "posting") submissionState = "waiting";
  }

  function markNetworkUnknown() {
    if (submissionState === "posting") submissionState = "unknown";
  }

  function markRejected() {
    submissionState = "idle";
    previousTaskStartedAt = null;
    trackedTaskStartedAt = null;
  }

  function observeTask(task, nodeState) {
    var startedAt = taskStartedAt(task);
    var newGeneration = false;
    var belongs = false;

    if (submissionState === "terminal" && trackedTaskStartedAt !== null &&
        startedAt !== null && startedAt !== trackedTaskStartedAt) {
      previousTaskStartedAt = null;
      trackedTaskStartedAt = null;
      submissionState = "idle";
      newGeneration = true;
    }

    if ((submissionState === "posting" || submissionState === "waiting" ||
         submissionState === "unknown") && startedAt !== null &&
        startedAt !== previousTaskStartedAt) {
      trackedTaskStartedAt = startedAt;
      newGeneration = true;
    } else if (submissionState === "idle" && !dirty &&
               nodeState === "relaunching" &&
               startedAt !== null && task && RELAUNCH_STEPS.has(task.current_step)) {
      trackedTaskStartedAt = startedAt;
      newGeneration = true;
    }

    if (trackedTaskStartedAt !== null && startedAt === trackedTaskStartedAt) {
      belongs = true;
      submissionState = task && TERMINAL_STEPS.has(task.current_step)
        ? "terminal"
        : "active";
    }

    return {
      belongs: belongs,
      new_generation: newGeneration,
      terminal: belongs && task ? TERMINAL_STEPS.has(task.current_step) : false,
      task: belongs ? task : null,
    };
  }

  function shouldConnectLogs(task) {
    var startedAt = taskStartedAt(task);
    if (trackedTaskStartedAt !== null) return startedAt === trackedTaskStartedAt;
    if (submissionState === "posting" || submissionState === "waiting" ||
        submissionState === "unknown") return false;
    return true;
  }

  return {
    reconcileRuntime: reconcileRuntime,
    setField: setField,
    reset: reset,
    validate: validate,
    preview: preview,
    beginSubmission: beginSubmission,
    markAccepted: markAccepted,
    markNetworkUnknown: markNetworkUnknown,
    markRejected: markRejected,
    observeTask: observeTask,
    shouldConnectLogs: shouldConnectLogs,
    isBusy: isBusy,
    state: function () {
      return {
        fields: Object.assign({}, fields),
        dirty: dirty,
        stale: stale,
        submission: submissionState,
        source_signature: sourceSignature,
        previous_task_started_at: previousTaskStartedAt,
        tracked_task_started_at: trackedTaskStartedAt,
      };
    },
  };
}
