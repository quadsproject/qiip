// Shared catalog-backed setup selection for the fleet and node-detail pages.

function formatInferenceEngine(value) {
  if (value === "vllm") return "vLLM";
  if (value === "llama_cpp") return "llama.cpp";
  return "—";
}

function formatRecommendationRuntime(value) {
  if (value === "mlx") return "MLX";
  if (value === "unknown") return "Unknown";
  return formatInferenceEngine(value);
}

function createSetupSelectionController(options) {
  options = options || {};
  var engineSelect = document.getElementById(
    options.engineSelectId || "setup-engine-select"
  );
  var modelSelect = document.getElementById(
    options.modelSelectId || "model-select"
  );
  var artifactSelect = document.getElementById(
    options.artifactSelectId || "artifact-select"
  );
  var status = document.getElementById(options.statusId || "model-status");
  var models = [];
  var artifacts = [];
  var catalogAvailable = false;
  var catalogWarnings = [];
  var preferred = null;
  var preferredApplied = false;
  var operatorChanged = false;
  var errorMessage = "Model catalog is still loading.";

  function appendOption(select, value, label, disabled) {
    var option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.disabled = Boolean(disabled);
    select.appendChild(option);
    return option;
  }

  engineSelect.textContent = "";
  appendOption(engineSelect, "vllm", "vLLM", false);
  appendOption(engineSelect, "llama_cpp", "llama.cpp", false);
  engineSelect.value = "vllm";

  function artifactLabel(artifact) {
    var alias = artifact.model_alias || artifact.repo_id;
    var entrypoint = artifact.entrypoint || "unknown GGUF";
    var revision = (artifact.resolved_revision || "").slice(0, 12);
    return alias + " — " + entrypoint + (revision ? " @" + revision : "");
  }

  function warningsFromCatalog(data) {
    var warnings = [];
    var incomplete = data.incomplete_count || 0;
    var unverifiable = data.unverifiable_count || 0;
    var invalidArtifacts = data.invalid_artifact_count || 0;
    var cacheWarnings = data.cache_warning_count || 0;
    if (unverifiable) {
      warnings.push(unverifiable + " cached model" +
        (unverifiable === 1 ? " lacks" : "s lack") +
        " manifest metadata and " +
        (unverifiable === 1 ? "was" : "were") +
        " hidden; re-download " +
        (unverifiable === 1 ? "it" : "them") + " to migrate the cache.");
    }
    if (incomplete) {
      warnings.push(incomplete + " incomplete cached model" +
        (incomplete === 1 ? " was" : "s were") +
        " hidden; re-download " +
        (incomplete === 1 ? "it" : "them") + ".");
    }
    if (invalidArtifacts) {
      warnings.push(invalidArtifacts + " invalid GGUF artifact" +
        (invalidArtifacts === 1 ? " was" : "s were") +
        " hidden; inspect gateway logs and cached GGUF files.");
    }
    if (cacheWarnings) {
      warnings.push(cacheWarnings + " cache warning" +
        (cacheWarnings === 1 ? " requires" : "s require") + " operator review");
    }
    return warnings;
  }

  function hasModel(repoId) {
    return models.some(function (model) { return model.repo_id === repoId; });
  }

  function hasArtifact(artifactId) {
    return artifacts.some(function (artifact) {
      return artifact.artifact_id === artifactId;
    });
  }

  function render() {
    var currentModel = modelSelect.value;
    var currentArtifact = artifactSelect.value;
    var engine = engineSelect.value || "vllm";
    if (preferredApplied && !operatorChanged && preferred) {
      engine = preferred.engine;
      engineSelect.value = engine;
    }

    modelSelect.textContent = "";
    for (var i = 0; i < models.length; i++) {
      appendOption(modelSelect, models[i].repo_id, models[i].repo_id, false);
    }
    artifactSelect.textContent = "";
    appendOption(artifactSelect, "", "Select a GGUF artifact", true);
    for (var j = 0; j < artifacts.length; j++) {
      appendOption(
        artifactSelect,
        artifacts[j].artifact_id,
        artifactLabel(artifacts[j]),
        false
      );
    }

    var desiredModel = currentModel;
    var desiredArtifact = currentArtifact;
    if (preferredApplied && !operatorChanged && preferred) {
      desiredModel = preferred.model || "";
      desiredArtifact = preferred.artifact_id || "";
    }

    if (desiredModel && hasModel(desiredModel)) {
      modelSelect.value = desiredModel;
    } else if (desiredModel && preferredApplied && !operatorChanged) {
      appendOption(
        modelSelect,
        desiredModel,
        "Previously selected: " + desiredModel + " (unavailable)",
        true
      );
      modelSelect.value = desiredModel;
    } else if (models.length > 0) {
      modelSelect.value = models[0].repo_id;
    }

    if (desiredArtifact && hasArtifact(desiredArtifact)) {
      artifactSelect.value = desiredArtifact;
    } else if (desiredArtifact && preferredApplied && !operatorChanged) {
      appendOption(
        artifactSelect,
        desiredArtifact,
        "Previously selected GGUF artifact (unavailable)",
        true
      );
      artifactSelect.value = desiredArtifact;
    } else {
      artifactSelect.value = "";
    }

    var llama = engine === "llama_cpp";
    modelSelect.style.display = !llama && models.length ? "" : "none";
    artifactSelect.style.display = llama && artifacts.length ? "" : "none";

    if (!catalogAvailable) {
      errorMessage = "Model catalog is unavailable.";
    } else if (llama && !hasArtifact(artifactSelect.value)) {
      errorMessage = desiredArtifact
        ? "The previously selected GGUF artifact is no longer available."
        : artifacts.length
          ? "Select a GGUF artifact."
          : "No GGUF files were discovered in the Hugging Face cache.";
    } else if (!llama && !hasModel(modelSelect.value)) {
      errorMessage = desiredModel
        ? "The previously selected vLLM model is no longer available."
        : "No verified models.";
    } else {
      errorMessage = "";
    }

    var messages = [];
    if (errorMessage) messages.push(errorMessage);
    messages = messages.concat(catalogWarnings);
    status.textContent = messages.join(" ");
    status.style.display = messages.length ? "inline" : "none";
  }

  function setCatalog(data) {
    models = data.models || [];
    artifacts = data.gguf_artifacts || [];
    catalogWarnings = warningsFromCatalog(data);
    catalogAvailable = true;
    render();
  }

  function setCatalogUnavailable() {
    models = [];
    artifacts = [];
    catalogWarnings = [];
    catalogAvailable = false;
    render();
  }

  function setPreferredNode(node) {
    if (operatorChanged || preferredApplied || !node || !node.engine) return;
    preferred = {
      engine: node.engine,
      model: node.engine === "vllm" ? (node.model || "") : "",
      artifact_id: node.engine === "llama_cpp" ? (node.artifact_id || "") : "",
    };
    preferredApplied = true;
    engineSelect.value = preferred.engine;
    render();
  }

  function getSelection() {
    if (!catalogAvailable) return null;
    if (engineSelect.value === "llama_cpp") {
      if (!hasArtifact(artifactSelect.value)) return null;
      return { engine: "llama_cpp", artifact_id: artifactSelect.value };
    }
    if (!hasModel(modelSelect.value)) return null;
    return { engine: "vllm", model: modelSelect.value };
  }

  function buildBody(base) {
    var selection = getSelection();
    if (!selection) return null;
    var body = {};
    Object.keys(base).forEach(function (key) { body[key] = base[key]; });
    Object.keys(selection).forEach(function (key) { body[key] = selection[key]; });
    return body;
  }

  function selectEngine(engine) {
    if (engine !== "vllm" && engine !== "llama_cpp") return;
    operatorChanged = true;
    engineSelect.value = engine;
    render();
  }

  engineSelect.addEventListener("change", function () {
    operatorChanged = true;
    render();
  });
  modelSelect.addEventListener("change", function () {
    operatorChanged = true;
  });
  artifactSelect.addEventListener("change", function () {
    operatorChanged = true;
  });

  render();

  return {
    setCatalog: setCatalog,
    setCatalogUnavailable: setCatalogUnavailable,
    setPreferredNode: setPreferredNode,
    selectEngine: selectEngine,
    getSelection: getSelection,
    buildBody: buildBody,
    isValid: function () { return getSelection() !== null; },
    errorMessage: function () { return errorMessage; },
  };
}
