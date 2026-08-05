// Config generators for OpenCode CLI, Pi coding agent, and OMP agent.
// Generators are pure functions testable via Node.js.

function generateOpenCodeConfig(baseUrl, modelId) {
  var base = baseUrl.replace(/\/+$/, "");
  return {
    $schema: "https://opencode.ai/config.json",
    provider: {
      qiip: {
        npm: "@ai-sdk/openai-compatible",
        name: "QIIP Inference Proxy",
        options: {
          baseURL: base + "/v1",
        },
        models: {
          [modelId]: {
            name: modelId,
          },
        },
      },
    },
    model: "qiip/" + modelId,
  };
}

function generatePiConfig(baseUrl, modelId) {
  var base = baseUrl.replace(/\/+$/, "");
  return {
    providers: {
      qiip: {
        baseUrl: base + "/v1",
        api: "openai-completions",
        apiKey: "none",
        compat: {
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
        },
        models: [{ id: modelId }],
      },
    },
  };
}

function generateOmpConfig(baseUrl, modelId) {
  var base = baseUrl.replace(/\/+$/, "");
  return [
    "providers:",
    "  qiip:",
    "    baseUrl: " + base + "/v1",
    "    apiKey: none",
    "    api: openai-completions",
    "    models:",
    "      - id: " + modelId,
  ].join("\n");
}

function downloadConfigFile(data, filename) {
  var isYaml = typeof data === "string";
  var content = isYaml ? data : JSON.stringify(data, null, 2);
  var blob = new Blob([content], { type: isYaml ? "text/yaml" : "application/json" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

var CONFIG_FORMATS = [
  { label: "OpenCode CLI", generator: generateOpenCodeConfig, filename: "opencode.json" },
  { label: "Pi Agent", generator: generatePiConfig, filename: "models.json" },
  { label: "OMP Agent", generator: generateOmpConfig, filename: "models.yaml" },
];

function createConfigDropdown(baseUrl, modelId, positionFn, onToggle) {
  var group = document.createElement("div");
  group.className = "action-group";

  var trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "btn-config";
  trigger.textContent = "Download ▾";

  var menu = document.createElement("div");
  menu.className = "action-menu";

  for (var i = 0; i < CONFIG_FORMATS.length; i++) {
    (function (fmt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-config";
      btn.textContent = fmt.label;
      btn.addEventListener("click", function () {
        downloadConfigFile(fmt.generator(baseUrl, modelId), fmt.filename);
        menu.classList.remove("open");
        if (onToggle) onToggle(false);
      });
      menu.appendChild(btn);
    })(CONFIG_FORMATS[i]);
  }

  trigger.addEventListener("click", function (e) {
    e.stopPropagation();
    var wasOpen = menu.classList.contains("open");
    document.querySelectorAll(".action-menu.open").forEach(function (m) { m.classList.remove("open"); });
    if (!wasOpen) {
      menu.classList.add("open");
      if (positionFn) positionFn(trigger, menu);
    }
    if (onToggle) onToggle(!wasOpen);
  });

  group.appendChild(trigger);
  group.appendChild(menu);
  return group;
}
