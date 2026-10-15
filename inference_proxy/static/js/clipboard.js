// Shared clipboard helpers: one copy convention for every page.
//
// qiipCopyToClipboard(value) writes *value* and resolves to a boolean. A
// string is written immediately; a promise for one is written through a
// promise-backed ClipboardItem, so the write starts inside the click's user
// activation even when the value arrives seconds later (a write issued after
// that window is refused with NotAllowedError). A rejected promise value
// rejects the returned promise instead, so the caller can report it; a
// promise that never settles leaves the write pending, so callers must give
// it one that settles (the page API helpers resolve or reject).
// qiipTokenCopyButton(options) builds the clipboard icon shown next to an
// abbreviated user token: it reveals the raw value through the page's own API
// helper, copies it, and keeps the value for a retry click when the browser
// refuses the write, so the secret never has to sit in the page.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  // Clipboard glyph drawn like the icons in the templates: 24x24, stroked
  // with currentColor so it follows the button's hover and disabled colors.
  var ICON_SHAPES = [
    [
      "path",
      {
        d: "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",
      },
    ],
    ["rect", { x: "8", y: "2", width: "8", height: "4", rx: "1", ry: "1" }],
  ];
  // A refused write keeps the fetched value for a retry click; after this
  // window the next click re-reveals, so a rotated token is never copied
  // twice.
  var RETRY_WINDOW_MS = 120000;

  function clipboardIcon() {
    var root = document.createElementNS(SVG_NS, "svg");
    root.setAttribute("viewBox", "0 0 24 24");
    root.setAttribute("fill", "none");
    root.setAttribute("stroke", "currentColor");
    root.setAttribute("stroke-width", "2");
    root.setAttribute("stroke-linecap", "round");
    root.setAttribute("stroke-linejoin", "round");
    root.setAttribute("aria-hidden", "true");
    ICON_SHAPES.forEach(function (shape) {
      var node = document.createElementNS(SVG_NS, shape[0]);
      Object.keys(shape[1]).forEach(function (key) {
        node.setAttribute(key, shape[1][key]);
      });
      root.appendChild(node);
    });
    return root;
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (err) {
        /* Insecure context or a denied permission: try the fallback. */
      }
    }
    var previous = document.activeElement;
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.select();
    var copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (err) {
      copied = false;
    }
    area.remove();
    // Selecting the offscreen textarea moved the focus; put it back.
    if (previous && previous.focus) previous.focus();
    return copied;
  }

  function isCopyable(text) {
    return typeof text === "string" && text !== "";
  }

  async function copyWhenReady(pending) {
    var text = await pending;
    if (!isCopyable(text)) return false;
    return copyText(text);
  }

  window.qiipCopyToClipboard = function (value) {
    if (typeof value === "string") {
      return isCopyable(value) ? copyText(value) : Promise.resolve(false);
    }
    var pending = Promise.resolve(value);
    if (window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
      var data = pending.then(function (text) {
        if (!isCopyable(text)) throw new Error("Nothing to copy.");
        return new Blob([text], { type: "text/plain" });
      });
      // ClipboardItem can throw before write() ever consumes the promise.
      data.catch(function () {});
      try {
        return navigator.clipboard.write([new ClipboardItem({ "text/plain": data })]).then(
          function () {
            return true;
          },
          function () {
            return copyWhenReady(pending);
          }
        );
      } catch (err) {
        return copyWhenReady(pending);
      }
    }
    return copyWhenReady(pending);
  };

  window.qiipTokenCopyButton = function (options) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = options.className || "token-copy";
    button.title = "Copy token";
    button.setAttribute("aria-label", "Copy token");
    button.appendChild(clipboardIcon());
    var retry = null; // {value, at}: a copy the browser refused, for a retry click
    // A value fetched while the tab was in focus is trusted only while it
    // stays in focus: another tab may have rotated the token since.
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) retry = null;
    });
    button.addEventListener("click", async function () {
      // Disabling the button blurs it in Chromium (and a keyboard activation
      // starts focused on it), so remember the trigger before that happens.
      var origin = document.activeElement === button ? button : null;
      button.disabled = true;
      try {
        var copied = false;
        if (retry && Date.now() - retry.at < RETRY_WINDOW_MS) {
          // A second click carries a fresh user activation; reuse the value
          // fetched by the first one instead of asking the server again.
          copied = await qiipCopyToClipboard(retry.value);
        } else {
          retry = null;
          copied = await qiipCopyToClipboard(
            options.reveal().then(function (payload) {
              if (!payload || typeof payload.token !== "string" || payload.token === "") {
                throw new Error("The server did not return a token.");
              }
              retry = { value: payload.token, at: Date.now() };
              return payload.token;
            })
          );
        }
        if (copied) {
          retry = null;
          options.notify("Token copied to clipboard.", "success");
        } else {
          options.notify(
            "Couldn't copy automatically. Press the copy icon again.",
            "warning"
          );
        }
      } catch (err) {
        options.notify(err.message, "error");
      } finally {
        button.disabled = false;
        // Restore the trigger only when the refusal left the focus nowhere
        // and the page still has it (never yank it from elsewhere, and never
        // pull the user back from outside the document).
        var active = document.activeElement;
        if (origin && document.hasFocus() && (!active || active === document.body)) {
          origin.focus();
        }
      }
    });
    return button;
  };
})();
