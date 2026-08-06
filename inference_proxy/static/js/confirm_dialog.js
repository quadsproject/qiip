// ponytail: one native <dialog>, reused for every confirmation, no framework needed.
// Replaces window.confirm() so destructive-action prompts match the app's theme and
// don't block the render thread with an unstyled browser modal.
// Built with createElement/appendChild (not innerHTML) to keep chat.js the sole
// sanitized-HTML sink in this codebase (see test_javascript_has_one_sanitized_html_sink).

(function () {
  var dialogEl = null;

  function buildDialog() {
    var dialog = document.createElement("dialog");
    dialog.className = "confirm-dialog";

    var body = document.createElement("div");
    body.className = "confirm-dialog-body";

    var title = document.createElement("h2");
    title.className = "confirm-dialog-title";

    var message = document.createElement("p");
    message.className = "confirm-dialog-message";

    var actions = document.createElement("div");
    actions.className = "confirm-dialog-actions";

    var cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "btn btn-neutral";

    var confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-primary";

    actions.appendChild(cancelBtn);
    actions.appendChild(confirmBtn);
    body.appendChild(title);
    body.appendChild(message);
    body.appendChild(actions);
    dialog.appendChild(body);
    document.body.appendChild(dialog);

    // Clicking the backdrop (a click landing on <dialog> itself, not a descendant) cancels.
    dialog.addEventListener("click", function (e) {
      if (e.target === dialog) dialog.dispatchEvent(new Event("cancel", { cancelable: true }));
    });

    return { dialog: dialog, title: title, message: message, cancelBtn: cancelBtn, confirmBtn: confirmBtn };
  }

  // confirmDialog({ title, message, confirmLabel, cancelLabel, danger }) -> Promise<boolean>
  window.confirmDialog = function (opts) {
    opts = opts || {};
    if (!dialogEl) dialogEl = buildDialog();
    var dialog = dialogEl.dialog;
    var cancelBtn = dialogEl.cancelBtn;
    var confirmBtn = dialogEl.confirmBtn;

    dialogEl.title.textContent = opts.title || "Confirm action";
    dialogEl.message.textContent = opts.message || "";
    cancelBtn.textContent = opts.cancelLabel || "Cancel";
    confirmBtn.textContent = opts.confirmLabel || "Confirm";
    confirmBtn.className = "btn " + (opts.danger ? "btn-danger" : "btn-primary");

    return new Promise(function (resolve) {
      function cleanup(result) {
        dialog.removeEventListener("cancel", onCancel);
        cancelBtn.removeEventListener("click", onCancelClick);
        confirmBtn.removeEventListener("click", onConfirmClick);
        dialog.close();
        resolve(result);
      }
      function onCancel(e) {
        e.preventDefault(); // we close explicitly in cleanup(), skip the native default close
        cleanup(false);
      }
      function onCancelClick() {
        cleanup(false);
      }
      function onConfirmClick() {
        cleanup(true);
      }
      dialog.addEventListener("cancel", onCancel); // fires on Escape, and on our synthetic backdrop click
      cancelBtn.addEventListener("click", onCancelClick);
      confirmBtn.addEventListener("click", onConfirmClick);
      dialog.showModal();
      cancelBtn.focus(); // safer default focus for destructive actions
    });
  };
})();
