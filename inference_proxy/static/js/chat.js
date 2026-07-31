// ponytail: vanilla fetch + ReadableStream SSE, no EventSource (POST required)
"use strict";

function showToast(message, type) {
  var container = document.getElementById("toast-container");
  var toast = document.createElement("div");
  toast.className = "toast toast-" + (type || "info");
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(function () { toast.classList.add("toast-visible"); });
  setTimeout(function () {
    toast.classList.remove("toast-visible");
    setTimeout(function () { toast.remove(); }, 300);
  }, 4000);
}

var messages = [];
var messageArea;
var messageAreaInner;
var emptyState;
var chatInput;
var sendBtn;
var modelSelect;
var streaming = false;
var modelsAvailable = false;
var systemPromptTextarea;
var systemPromptToggle;

// Assistant Markdown is intentionally limited to inert formatting. Links render
// as plain text and images/media/forms are removed, so model output cannot
// trigger an outbound request or create an interactive control.
var ASSISTANT_MARKDOWN_SANITIZE_CONFIG = {
  ALLOWED_TAGS: [
    "p", "br", "strong", "em", "del", "blockquote", "code", "pre",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "table", "thead", "tbody", "tr", "th", "td",
  ],
  ALLOWED_ATTR: [],
  ALLOW_ARIA_ATTR: false,
  ALLOW_DATA_ATTR: false,
};

function renderAssistantMarkdown(target, content) {
  target.innerHTML = DOMPurify.sanitize(
    marked.parse(content || ""),
    ASSISTANT_MARKDOWN_SANITIZE_CONFIG,
  );
}

function isNearBottom(el, threshold) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function hideEmptyState() {
  if (emptyState && emptyState.style.display !== "none") {
    emptyState.style.display = "none";
  }
}

function addMessage(role, content) {
  hideEmptyState();
  var bubble = document.createElement("div");
  bubble.className = "message-bubble bubble-" + role;
  if (role === "user") {
    bubble.textContent = content;
  } else {
    renderAssistantMarkdown(bubble, content);
  }
  messageAreaInner.appendChild(bubble);
  messageArea.scrollTop = messageArea.scrollHeight;
  return bubble;
}

function setInputEnabled(enabled) {
  sendBtn.disabled = !enabled || !modelsAvailable;
  chatInput.disabled = !enabled;
}

async function streamResponse(bubble) {
  var rawText = "";
  var cursor = document.createElement("span");
  cursor.className = "streaming-cursor";
  bubble.appendChild(cursor);

  try {
    var resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(function () {
        var payloadMessages = messages.slice();
        var sp = systemPromptTextarea ? systemPromptTextarea.value.trim() : "";
        if (sp) {
          payloadMessages.unshift({ role: "system", content: sp });
        }
        return { model: modelSelect.value, messages: payloadMessages, stream: true };
      }()),
    });

    if (!resp.ok) {
      var errData = await resp.json().catch(function () { return {}; });
      var errMsg = (errData.error && errData.error.message) || "Request failed: HTTP " + resp.status;
      bubble.textContent = errMsg;
      bubble.classList.add("bubble-error");
      return;
    }

    var reader = resp.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";

    while (true) {
      var result = await reader.read();
      if (result.done) break;

      buffer += decoder.decode(result.value, { stream: true });
      var lines = buffer.split("\n");
      // ponytail: last element may be incomplete line, keep in buffer
      buffer = lines.pop();

      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (!line.startsWith("data: ")) continue;
        var data = line.slice(6);
        if (data === "[DONE]") continue;

        try {
          var parsed = JSON.parse(data);
          if (parsed.error && parsed.error.message) {
            bubble.textContent = parsed.error.message;
            bubble.classList.add("bubble-error");
            return;
          }
          var delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
          if (delta && delta.content) {
            rawText += delta.content;
            var shouldScroll = isNearBottom(messageArea, 40);
            renderAssistantMarkdown(bubble, rawText);
            bubble.appendChild(cursor);
            if (shouldScroll) {
              messageArea.scrollTop = messageArea.scrollHeight;
            }
          }
        } catch (e) {
          // ponytail: skip malformed SSE chunks silently
        }
      }
    }

    // finalize
    renderAssistantMarkdown(bubble, rawText);
    messages.push({ role: "assistant", content: rawText });
  } catch (err) {
    showToast("Could not reach the server. Check your connection.", "error");
    if (!rawText) {
      bubble.textContent = "Connection error";
      bubble.classList.add("bubble-error");
    }
  } finally {
    var cur = bubble.querySelector(".streaming-cursor");
    if (cur) cur.remove();
    streaming = false;
    setInputEnabled(true);
    chatInput.focus();
  }
}

function sendMessage() {
  var text = chatInput.value.trim();
  if (!text || streaming || !modelSelect.value) return;
  streaming = true;

  messages.push({ role: "user", content: text });
  addMessage("user", text);
  chatInput.value = "";
  chatInput.style.height = "auto";
  setInputEnabled(false);

  var bubble = addMessage("assistant", "");
  streamResponse(bubble);
}

async function loadModels() {
  try {
    var resp = await fetch("/v1/models");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var data = await resp.json();
    var models = data.data || [];
    modelSelect.textContent = "";
    modelsAvailable = models.length > 0;

    if (models.length === 0) {
      var opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No models available";
      opt.disabled = true;
      opt.selected = true;
      modelSelect.appendChild(opt);
      setInputEnabled(!streaming);
      return;
    }

    for (var i = 0; i < models.length; i++) {
      var opt = document.createElement("option");
      opt.value = models[i].id;
      opt.textContent = models[i].id;
      modelSelect.appendChild(opt);
    }
    setInputEnabled(!streaming);
  } catch (e) {
    modelSelect.textContent = "";
    modelsAvailable = false;
    var opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No models available";
    opt.disabled = true;
    opt.selected = true;
    modelSelect.appendChild(opt);
    setInputEnabled(!streaming);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  messageArea = document.getElementById("message-area");
  messageAreaInner = messageArea.querySelector(".message-area-inner");
  emptyState = document.getElementById("empty-state");
  chatInput = document.getElementById("chat-input");
  sendBtn = document.getElementById("send-btn");
  modelSelect = document.getElementById("model-select");
  systemPromptTextarea = document.getElementById("system-prompt");
  systemPromptToggle = document.querySelector(".system-prompt-toggle");

  // ponytail: restore saved system prompt from localStorage
  var savedPrompt = localStorage.getItem("systemPrompt");
  if (savedPrompt !== null && systemPromptTextarea) {
    systemPromptTextarea.value = savedPrompt;
  }

  if (systemPromptToggle) {
    systemPromptToggle.addEventListener("click", function () {
      var expanded = this.getAttribute("aria-expanded") === "true";
      this.setAttribute("aria-expanded", expanded ? "false" : "true");
      document.getElementById("system-prompt-panel").classList.toggle("expanded");
    });
  }

  if (systemPromptTextarea) {
    systemPromptTextarea.addEventListener("input", function () {
      localStorage.setItem("systemPrompt", this.value);
    });
  }

  loadModels();

  sendBtn.addEventListener("click", sendMessage);

  chatInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ponytail: auto-grow textarea, capped at 200px via CSS max-height
  chatInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
  });
});
