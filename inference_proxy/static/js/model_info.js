// Optional information next to model selectors; never changes selection.
(function () {
  "use strict";
  var tip = document.createElement("div");
  tip.id = "ob-model-tooltip";
  tip.className = "ob-model-tooltip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.appendChild(tip);
  var active = null, pinned = false, timer = null;

  function close() {
    clearTimeout(timer);
    if (active) active.removeAttribute("aria-describedby");
    active = null;
    pinned = false;
    tip.hidden = true;
  }
  function leave() {
    clearTimeout(timer);
    if (!pinned && active !== document.activeElement) timer = setTimeout(close, 150);
  }
  tip.addEventListener("mouseenter", function () { clearTimeout(timer); });
  tip.addEventListener("mouseleave", leave);
  document.addEventListener("click", function (event) {
    if (active && !active.contains(event.target) && !tip.contains(event.target)) close();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && active) {
      close();
      event.preventDefault();
      event.stopPropagation();
    }
  });
  window.addEventListener("resize", close);
  window.addEventListener("scroll", close, true);

  function displayName(model) {
    // Publisher and GGUF packaging are not part of the user-facing name.
    return model.split("/").pop().replace(/(?:-MTP)?-GGUF$/i, "") || model;
  }

  function wrap(button, model, details) {
    if (!details) return button;
    var row = document.createElement("div");
    row.className = "ob-model-option";
    var info = document.createElement("button");
    info.type = "button";
    info.className = "ob-model-info";
    info.textContent = "ⓘ";
    info.setAttribute("aria-label", "Information about " + displayName(model));
    function open() {
      clearTimeout(timer);
      if (active !== info) close();
      active = info;
      info.setAttribute("aria-describedby", tip.id);
      var text = document.createElement("p");
      text.textContent = "Inputs: " + details.input_modalities.join(", ");
      var context = document.createElement("p");
      context.textContent = "Context: " + Number(details.context_tokens).toLocaleString("en-US") + " tokens";
      tip.replaceChildren(text, context);
      tip.hidden = false;
      var rect = info.getBoundingClientRect();
      var bounds = tip.getBoundingClientRect();
      tip.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - bounds.width - 8)) + "px";
      tip.style.top = Math.max(8, rect.bottom + bounds.height + 8 <= window.innerHeight
        ? rect.bottom + 8 : rect.top - bounds.height - 8) + "px";
    }
    info.addEventListener("mouseenter", open);
    info.addEventListener("mouseleave", leave);
    info.addEventListener("focus", open);
    info.addEventListener("blur", close);
    info.addEventListener("click", function (event) {
      event.stopPropagation();
      if (active === info && pinned) close();
      else { open(); pinned = true; }
    });
    row.append(button, info);
    return row;
  }
  function sizeChoices(grid) {
    grid.style.removeProperty("--ob-model-width");
    var width = 0;
    grid.querySelectorAll("button[data-model]").forEach(function (button) {
      width = Math.max(width, button.getBoundingClientRect().width);
    });
    grid.style.setProperty("--ob-model-width", Math.ceil(width) + "px");
  }
  window.QiipModelInfo = { wrap: wrap, close: close, displayName: displayName, sizeChoices: sizeChoices };
})();
