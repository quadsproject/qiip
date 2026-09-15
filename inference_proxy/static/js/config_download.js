// Config generators for OpenCode CLI, Pi coding agent, and OMP agent.
// Generators are pure functions testable via Node.js.
//
// *opts* carries node display info: `name` (operator-facing name) and
// `admin_only` (admin-only servers require a bearer token, so the generated
// config declares apiKey auth with a placeholder instead of `auth: none` —
// an admin-only server is unreachable anonymously by design).

var TOKEN_PLACEHOLDER = "<paste-qiip-token-here>";

function configApiKey(opts) {
  // Admin-only servers: a real minted token when the download flow obtained one,
  // otherwise an explicit placeholder (never silently `auth: none`).
  if (opts && opts.admin_only) {
    return opts.token || TOKEN_PLACEHOLDER;
  }
  return null;
}

function generateOpenCodeConfig(baseUrl, modelId, opts) {
  var base = baseUrl.replace(/\/+$/, "");
  var options = {
    baseURL: base + "/v1",
  };
  var apiKey = configApiKey(opts);
  if (apiKey) {
    options.apiKey = apiKey;
  }
  return {
    $schema: "https://opencode.ai/config.json",
    provider: {
      qiip: {
        npm: "@ai-sdk/openai-compatible",
        name: "QIIP Inference Proxy",
        options: options,
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

function generatePiConfig(baseUrl, modelId, opts) {
  var base = baseUrl.replace(/\/+$/, "");
  // configApiKey already returns the placeholder for admin_only servers
  // without a token, so no re-derivation is needed here.
  var apiKeyValue = configApiKey(opts) || "none";
  return {
    providers: {
      qiip: {
        baseUrl: base + "/v1",
        api: "openai-completions",
        apiKey: apiKeyValue,
        compat: {
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
        },
        models: [{ id: modelId }],
      },
    },
  };
}

function yamlScalar(v) {
  if (/: | #|[{}\[\]]/.test(v)) {
    return '"' + v.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
  }
  return v;
}

function generateOmpConfig(baseUrl, modelId, opts) {
  var base = baseUrl.replace(/\/+$/, "");
  var displayName = opts && opts.name ? opts.name : modelId + " (qiip)";
  var lines = [
    "providers:",
    "  qiip:",
    "    baseUrl: " + yamlScalar(base + "/v1"),
  ];
  if (opts && opts.admin_only) {
    // Admin-only inference servers are reachable only with an admin-role apiKey.
    // configApiKey returns the placeholder when the download flow has no token.
    var apiKey = configApiKey(opts);
    lines.push("    auth: apiKey");
    lines.push("    apiKey: " + yamlScalar(apiKey));
  } else {
    lines.push("    auth: none");
  }
  lines.push("    api: openai-completions");
  lines.push("    models:");
  lines.push("      - id: " + yamlScalar(modelId));
  lines.push("        name: " + yamlScalar(displayName));
  return lines.join("\n");
}

function downloadConfigFile(data, filename) {
  var isYaml = typeof data === "string";
  var content = isYaml ? data : JSON.stringify(data, null, 2);
  var blob = new Blob([content], { type: isYaml ? "application/yaml" : "application/json" });
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

function createConfigDropdown(baseUrl, modelId, positionFn, onToggle, opts) {
  var group = document.createElement("div");
  group.className = "action-group";

  var trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "btn btn-sm btn-neutral btn-trigger";
  trigger.textContent = "Download";

  var menu = document.createElement("div");
  menu.className = "action-menu";

  for (var i = 0; i < CONFIG_FORMATS.length; i++) {
    (function (fmt) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-sm btn-neutral";
      btn.textContent = fmt.label;
      btn.addEventListener("click", async function () {
        var generatorOpts = opts || {};
        if (generatorOpts.admin_only) {
          // Admin-only servers need a bearer token: share the user's single
          // agent-config key (minted on first use, then reused). It is
          // derived server-side and never stored, so every download of any
          // admin-only server -- any browser, any machine -- embeds the same
          // key. A revoke rotates it; the next download gets the new one.
          // Minting requires a Google-user session (/profile/tokens); a
          // local-admin/Basic identity cannot mint, so abort the download
          // rather than shipping a knowingly unusable placeholder config.
          try {
            var mintResp = await fetch("/profile/tokens", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: "agent-config" }),
            });
            if (mintResp.ok) {
              var created = await mintResp.json();
              generatorOpts = Object.assign({}, generatorOpts, {
                token: created.token,
              });
            } else {
              var mintErr = await mintResp.json().catch(function () {
                return { detail: "HTTP " + mintResp.status };
              });
              if (typeof window.showToast === "function") {
                window.showToast(
                  "Cannot download admin_only server config: " +
                    (mintErr.detail || "HTTP error") +
                    ". Sign in with Google to mint an agent-config token.",
                  "error"
                );
              }
              menu.classList.remove("open");
              if (onToggle) onToggle(false);
              return;
            }
          } catch (err) {
            if (typeof window.showToast === "function") {
              window.showToast("Token fetch failed: " + err.message, "error");
            }
            menu.classList.remove("open");
            if (onToggle) onToggle(false);
            return;
          }
        }
        downloadConfigFile(fmt.generator(baseUrl, modelId, generatorOpts), fmt.filename);
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
