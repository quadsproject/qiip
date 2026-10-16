// ponytail: vanilla fetch + ReadableStream SSE, no EventSource (POST required)
"use strict";

var messages = [];
var messageArea;
var messageAreaInner;
var emptyState;
var chatInput;
var sendBtn;
var modelSelect;
var streaming = false;
var modelsAvailable = false;
var authToken = null;
var chatDisabled = false;
var systemPromptTextarea;
var systemPromptToggle;

// Every /v1 call carries the playground credential when the viewer has one: a
// signed-in normal user's own token, so their usage is recorded against it and
// the model list is limited to what that token may use. The request also asks
// for strict authentication, so a token that no longer resolves is refused
// instead of silently becoming an anonymous call.
function authHeaders(headers) {
  if (authToken) {
    headers["Authorization"] = "Bearer " + authToken;
    headers["X-Qiip-Require-Token"] = "1";
  }
  return headers;
}

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
  sendBtn.disabled = !enabled || !modelsAvailable || chatDisabled;
  chatInput.disabled = !enabled || chatDisabled;
}

// Bound what gets re-sent: one long answer can fill the model's context and
// every later turn would otherwise re-send the same oversized transcript.
// Approximate budget (4 chars per token); the newest turn is always kept.
var MAX_HISTORY_CHARS = 24000;

function trimHistory() {
  var start = Math.max(0, messages.length - 2);
  var total = 0;
  for (var i = start; i >= 2; i -= 2) {
    var pair = messages[i - 2].content.length + messages[i - 1].content.length;
    if (total + pair > MAX_HISTORY_CHARS) break;
    total += pair;
    start = i - 2;
  }
  if (start > 0) messages.splice(0, start);
}

function commitTurn(userMessage, assistantContent) {
  messages.push(userMessage, { role: "assistant", content: assistantContent });
  trimHistory();
}

function extractedErrorMessage(errorData) {
  var nested = errorData && errorData.error && errorData.error.message;
  var flat = errorData && errorData.message;
  var message = nested || flat;
  return typeof message === "string" && message ? message : null;
}

function errorMessage(errorData, status) {
  return extractedErrorMessage(errorData) || "Request failed: HTTP " + status;
}

function systemPromptGuidance(message) {
  var prompt = systemPromptTextarea ? systemPromptTextarea.value.trim() : "";
  if (prompt && message.toLowerCase().includes("conversation roles must alternate")) {
    return message + "\n\nThis model may not accept a system role. Clear the System Prompt and try again.";
  }
  return message;
}

async function streamResponse(bubble, pendingUser) {
  var rawText = "";
  var cursor = document.createElement("span");
  cursor.className = "streaming-cursor";
  bubble.appendChild(cursor);

  // Reveal per turn instead of reusing the token loaded with the page: a token
  // replaced in another /start tab is followed, and a revoked one is refused
  // before anything is sent.
  var fresh = await revealCredential();
  if (fresh === null) {
    // The conversation stays visible, so the way out has to travel with it.
    bubble.textContent = credentialError + " ";
    bubble.classList.add("bubble-error");
    appendStartLink(bubble);
    failClosed();
    return;
  }
  if (fresh) authToken = fresh;

  try {
    var resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(function () {
        var payloadMessages = messages.concat([pendingUser]);
        var sp = systemPromptTextarea ? systemPromptTextarea.value.trim() : "";
        if (sp) {
          payloadMessages.unshift({ role: "system", content: sp });
        }
        return {
          model: modelSelect.value,
          messages: payloadMessages,
          stream: true,
          // vLLM omits usage from a stream unless it is requested; without it
          // the gateway can only record zeros for this turn.
          stream_options: { include_usage: true },
        };
      }()),
    });

    if (!resp.ok) {
      var errData = await resp.json().catch(function () { return {}; });
      bubble.textContent = systemPromptGuidance(errorMessage(errData, resp.status));
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
          var streamError = extractedErrorMessage(parsed);
          if (streamError) {
            bubble.textContent = systemPromptGuidance(streamError);
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
    // A protocol-successful empty response is still a completed assistant turn.
    // Keeping the pair preserves the backend result in future context.
    commitTurn(pendingUser, rawText);
  } catch (err) {
    showToast("Could not reach the server. Check your connection.", "error");
    if (rawText) {
      // Preserve tokens the user saw so the next request has the same context.
      commitTurn(pendingUser, rawText);
    } else {
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
  if (!text || streaming || chatDisabled || !modelSelect.value) return;
  streaming = true;

  var pendingUser = { role: "user", content: text };
  addMessage("user", text);
  chatInput.value = "";
  chatInput.style.height = "auto";
  setInputEnabled(false);

  var bubble = addMessage("assistant", "");
  return streamResponse(bubble, pendingUser);
}

async function loadModels() {
  try {
    var resp = await fetch("/v1/models", { headers: authHeaders({}) });
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

// A notice means this viewer has no token to chat with: the playground stays
// disabled instead of making unattributed anonymous requests.
function showChatNotice(message) {
  failClosed();

  var placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Chat unavailable";
  placeholder.disabled = true;
  placeholder.selected = true;
  modelSelect.textContent = "";
  modelSelect.appendChild(placeholder);

  var paragraph = document.getElementById("empty-state-text");
  if (paragraph) {
    paragraph.textContent = message + " ";
    appendStartLink(paragraph);
  }
}

// The way out of a disabled playground is always the same page.
function appendStartLink(target) {
  var link = document.createElement("a");
  link.href = "/start";
  link.textContent = "Open /start";
  target.appendChild(link);
}

var CREDENTIAL_FAILURE =
  "Could not load your chat credential. Reload the page and try again.";

var credentialError = null;

function credentialNotice(detail) {
  return typeof detail === "string" && detail ? detail : CREDENTIAL_FAILURE;
}

function credentialRequired() {
  return !(
    typeof ANONYMOUS_PLAYGROUND !== "undefined" &&
    ANONYMOUS_PLAYGROUND === true
  );
}

// The page shell says whether this viewer chats anonymously (admin role, the
// full-access trust list, or the local admin): none of them has a personal
// token to reveal. Every other viewer runs the playground on their own token.
//
// Returns the revealed token, "" for the anonymous playground, or null when
// the reveal failed (the reason is in credentialError).
async function revealCredential() {
  credentialError = null;
  if (!credentialRequired()) return "";
  var resp = null;
  var body = {};
  try {
    resp = await fetch("/onboarding/token/reveal", { method: "POST" });
    body = await resp.json();
  } catch (e) {
    credentialError = CREDENTIAL_FAILURE;
    return null;
  }
  if (!resp.ok) {
    credentialError = credentialNotice(body.detail);
    return null;
  }
  if (!body.token) {
    credentialError = CREDENTIAL_FAILURE;
    return null;
  }
  return body.token;
}

// A viewer who must be attributed never falls back to anonymous calls.
function failClosed() {
  chatDisabled = true;
  modelsAvailable = false;
  setInputEnabled(false);
}

async function loadCredential() {
  var token = await revealCredential();
  if (token === null) {
    showChatNotice(credentialError);
    return;
  }
  if (token) authToken = token;
  loadModels();
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
    if (savedPrompt.trim() && systemPromptToggle) {
      systemPromptToggle.setAttribute("aria-expanded", "true");
      document.getElementById("system-prompt-panel").classList.add("expanded");
    }
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

  loadCredential();

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
