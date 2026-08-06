"""Behavioral tests for the managed llama.cpp relaunch controller."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER_JS = _ROOT / "inference_proxy/static/js/llamacpp_relaunch.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const sandbox = { console };
vm.createContext(sandbox);
vm.runInContext(source, sandbox);

function runtime(requested, overrides) {
  overrides = overrides || {};
  return {
    requested,
    effective: Object.assign({
      train_context: 262144,
      context_per_slot: 12544,
      slot_context_limit: 12544,
      slots: 1,
      aggregate_context: 12544,
      cache_type_k: "q8_0",
      cache_type_v: "q8_0",
      flash_attn: "on",
      kv_unified: true,
      gpu_layers: 31,
      total_layers: 31,
    }, overrides.effective || {}),
    gpus: overrides.gpus || [
      { index: 0, total_mib: 15360, used_mib: 14117, free_mib: 1243 },
    ],
    observed_at: overrides.observed_at || "2026-08-05T21:55:24Z",
  };
}
"""


def _run_controller(scenario: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for llama.cpp relaunch JavaScript tests")
    result = subprocess.run(
        [node, "-e", f"{_HARNESS}\n{scenario}", str(_CONTROLLER_JS)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    parsed: object = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_untouched_automatic_policy_roundtrips_non_default_reserve() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime({ sizing: "auto", fit_target_mib: 1024 }));
process.stdout.write(JSON.stringify({
  state: controller.state(),
  validation: controller.validate(),
}));
"""
    )

    assert result["state"]["fields"] == {
        "sizing": "auto",
        "fit_target_mib": "1024",
        "context_per_slot": "12544",
        "slots": "1",
        "cache_type": "q8_0",
    }
    assert result["state"]["dirty"] is False
    assert result["validation"]["body"] == {
        "sizing": "auto",
        "fit_target_mib": 1024,
    }


def test_untouched_custom_policy_roundtrips_exactly() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
const requested = {
  sizing: "custom", fit_target_mib: 768,
  context_per_slot: 32768, slots: 3, cache_type: "f16",
};
controller.reconcileRuntime(runtime(requested));
process.stdout.write(JSON.stringify({
  dirty: controller.state().dirty,
  body: controller.validate().body,
}));
"""
    )

    assert result == {
        "dirty": False,
        "body": {
            "sizing": "custom",
            "fit_target_mib": 768,
            "context_per_slot": 32768,
            "slots": 3,
            "cache_type": "f16",
        },
    }


def test_switching_automatic_policy_to_custom_seeds_effective_values() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime({ sizing: "auto", fit_target_mib: 512 }));
controller.setField("sizing", "custom");
process.stdout.write(JSON.stringify({
  fields: controller.state().fields,
  body: controller.validate().body,
}));
"""
    )

    assert result["fields"]["context_per_slot"] == "12544"
    assert result["fields"]["slots"] == "1"
    assert result["fields"]["cache_type"] == "q8_0"
    assert result["body"] == {
        "sizing": "custom",
        "fit_target_mib": 512,
        "context_per_slot": 12544,
        "slots": 1,
        "cache_type": "q8_0",
    }


def test_polling_reseeds_pristine_form_but_marks_dirty_form_stale() -> None:
    result = _run_controller(
        r"""
const first = runtime({ sizing: "auto", fit_target_mib: 512 });
const second = runtime(
  { sizing: "auto", fit_target_mib: 1024 },
  { observed_at: "2026-08-05T22:00:00Z" }
);
const pristine = sandbox.createLlamaCppRelaunchController();
pristine.reconcileRuntime(first);
const pristineOutcome = pristine.reconcileRuntime(second);

const dirty = sandbox.createLlamaCppRelaunchController();
dirty.reconcileRuntime(first);
dirty.setField("fit_target_mib", "768");
const dirtyOutcome = dirty.reconcileRuntime(second);
process.stdout.write(JSON.stringify({
  pristineOutcome,
  pristine: pristine.state(),
  dirtyOutcome,
  dirty: dirty.state(),
}));
"""
    )

    assert result["pristineOutcome"] == "seeded"
    assert result["pristine"]["fields"]["fit_target_mib"] == "1024"
    assert result["pristine"]["dirty"] is False
    assert result["dirtyOutcome"] == "stale"
    assert result["dirty"]["fields"]["fit_target_mib"] == "768"
    assert result["dirty"]["stale"] is True


def test_reset_adopts_latest_runtime_after_stale_edit() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime({ sizing: "auto", fit_target_mib: 512 }));
controller.setField("fit_target_mib", "768");
controller.reconcileRuntime(runtime(
  { sizing: "auto", fit_target_mib: 1024 },
  { observed_at: "2026-08-05T22:00:00Z" }
));
controller.reset();
process.stdout.write(JSON.stringify(controller.state()));
"""
    )

    assert result["fields"]["fit_target_mib"] == "1024"
    assert result["dirty"] is False
    assert result["stale"] is False


@pytest.mark.parametrize(
    ("context", "slots", "valid"),
    [
        (255, 1, False),
        (256, 1, True),
        (16_843_008, 255, True),
        (16_843_264, 255, False),
        (256, 256, True),
        (256, 257, False),
    ],
)
def test_custom_boundaries_match_the_gateway_contract(
    context: int,
    slots: int,
    valid: bool,
) -> None:
    result = _run_controller(
        f"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime(
  {{
    sizing: "custom", fit_target_mib: 512,
    context_per_slot: {context}, slots: {slots}, cache_type: "q8_0",
  }},
  {{ effective: {{ train_context: 20000000 }} }}
));
process.stdout.write(JSON.stringify(controller.validate()));
"""
    )

    assert result["valid"] is valid


def test_fit_target_must_be_below_every_gpu_total() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime(
  { sizing: "auto", fit_target_mib: 512 },
  { gpus: [
    { index: 0, total_mib: 15360, used_mib: 14000, free_mib: 1360 },
    { index: 1, total_mib: 1024, used_mib: 512, free_mib: 512 },
  ] }
));
controller.setField("fit_target_mib", "1024");
process.stdout.write(JSON.stringify(controller.validate()));
"""
    )

    assert result["valid"] is False
    assert "smaller than every GPU" in result["errors"]["fit_target_mib"]


def test_started_at_identifies_fast_terminal_relaunch_without_intermediate_poll() -> (
    None
):
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.beginSubmission({
  started_at: "2026-08-05T20:00:00Z", current_step: "complete",
});
controller.markAccepted();
const oldTask = controller.observeTask({
  started_at: "2026-08-05T20:00:00Z", current_step: "complete",
}, "healthy");
const failedTask = controller.observeTask({
  started_at: "2026-08-05T21:00:00Z", current_step: "failed",
  failed_step: "relaunch_validating", error: "artifact disappeared",
}, "healthy");
process.stdout.write(JSON.stringify({ oldTask, failedTask, state: controller.state() }));
"""
    )

    assert result["oldTask"]["belongs"] is False
    assert result["failedTask"]["belongs"] is True
    assert result["failedTask"]["new_generation"] is True
    assert result["failedTask"]["terminal"] is True
    assert result["state"]["submission"] == "terminal"


def test_preexisting_failed_task_is_not_adopted_for_healthy_node() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
const observation = controller.observeTask({
  started_at: "2026-08-05T20:00:00Z", current_step: "failed",
  failed_step: "starting_llamacpp",
}, "healthy");
process.stdout.write(JSON.stringify({ observation, state: controller.state() }));
"""
    )

    assert result["observation"]["belongs"] is False
    assert result["state"]["submission"] == "idle"


def test_dirty_form_does_not_adopt_another_tabs_relaunch() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
controller.reconcileRuntime(runtime({ sizing: "auto", fit_target_mib: 512 }));
controller.setField("fit_target_mib", "768");
const observation = controller.observeTask({
  started_at: "2026-08-05T21:00:00Z", current_step: "draining",
}, "relaunching");
const outcome = controller.reconcileRuntime(runtime(
  { sizing: "custom", fit_target_mib: 1024,
    context_per_slot: 8192, slots: 2, cache_type: "f16" },
  { observed_at: "2026-08-05T22:00:00Z" }
));
process.stdout.write(JSON.stringify({ observation, outcome, state: controller.state() }));
"""
    )

    assert result["observation"]["belongs"] is False
    assert result["outcome"] == "stale"
    assert result["state"]["fields"]["fit_target_mib"] == "768"
    assert result["state"]["stale"] is True


def test_double_submission_is_rejected_until_the_lifecycle_finishes() -> None:
    result = _run_controller(
        r"""
const controller = sandbox.createLlamaCppRelaunchController();
const first = controller.beginSubmission(null);
const second = controller.beginSubmission(null);
controller.markAccepted();
const third = controller.beginSubmission(null);
process.stdout.write(JSON.stringify({ first, second, third, state: controller.state() }));
"""
    )

    assert result["first"] is True
    assert result["second"] is False
    assert result["third"] is False
    assert result["state"]["submission"] == "waiting"


def test_log_connection_waits_for_the_new_task_generation() -> None:
    result = _run_controller(
        r"""
const oldTask = {
  started_at: "2026-08-05T20:00:00Z", current_step: "complete",
};
const newTask = {
  started_at: "2026-08-05T21:00:00Z", current_step: "relaunch_validating",
};
const controller = sandbox.createLlamaCppRelaunchController();
controller.beginSubmission(oldTask);
controller.markAccepted();
const oldConnect = controller.shouldConnectLogs(oldTask);
const observation = controller.observeTask(newTask, "healthy");
const newConnect = controller.shouldConnectLogs(newTask);
process.stdout.write(JSON.stringify({ oldConnect, observation, newConnect }));
"""
    )

    assert result["oldConnect"] is False
    assert result["observation"]["new_generation"] is True
    assert result["newConnect"] is True


def test_later_task_releases_terminal_relaunch_generation_for_logs() -> None:
    result = _run_controller(
        r"""
const oldTask = {
  started_at: "2026-08-05T20:00:00Z", current_step: "complete",
};
const relaunchTask = {
  started_at: "2026-08-05T21:00:00Z", current_step: "relaunch_validating",
};
const completedRelaunch = {
  started_at: "2026-08-05T21:00:00Z", current_step: "complete",
};
const teardownTask = {
  started_at: "2026-08-05T22:00:00Z", current_step: "stopping_vllm",
};
const controller = sandbox.createLlamaCppRelaunchController();
controller.beginSubmission(oldTask);
controller.markAccepted();
controller.observeTask(relaunchTask, "relaunching");
const terminal = controller.observeTask(completedRelaunch, "healthy");
const teardown = controller.observeTask(teardownTask, "draining");
process.stdout.write(JSON.stringify({
  terminal,
  teardown,
  connect: controller.shouldConnectLogs(teardownTask),
  state: controller.state(),
}));
"""
    )

    assert result["terminal"]["terminal"] is True
    assert result["teardown"] == {
        "belongs": False,
        "new_generation": True,
        "terminal": False,
        "task": None,
    }
    assert result["connect"] is True
    assert result["state"]["submission"] == "idle"
    assert result["state"]["previous_task_started_at"] is None
    assert result["state"]["tracked_task_started_at"] is None


def test_ambiguous_network_outcome_stays_locked_without_a_new_task() -> None:
    result = _run_controller(
        r"""
const oldTask = {
  started_at: "2026-08-05T20:00:00Z", current_step: "complete",
};
const controller = sandbox.createLlamaCppRelaunchController();
controller.beginSubmission(oldTask);
controller.markNetworkUnknown();
const observation = controller.observeTask(oldTask, "healthy");
process.stdout.write(JSON.stringify({
  observation,
  connect: controller.shouldConnectLogs(oldTask),
  busy: controller.isBusy(),
  state: controller.state(),
}));
"""
    )

    assert result["observation"]["belongs"] is False
    assert result["observation"]["new_generation"] is False
    assert result["connect"] is False
    assert result["busy"] is True
    assert result["state"]["submission"] == "unknown"
