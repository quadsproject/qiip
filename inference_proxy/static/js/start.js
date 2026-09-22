// QIIP start page: onboarding wizard + token home for normal users.
// Vanilla JS, no framework. One step is on screen at a time; every node is
// built with textContent so server data can never inject markup.
(function () {
  "use strict";

  var stage = document.getElementById("ob-stage");
  var backBtn = document.getElementById("ob-back");
  var hint = document.getElementById("ob-hint");
  var bar = document.getElementById("ob-progress-bar");
  var toastEl = document.getElementById("ob-toast");

  var data = null; // /onboarding/state payload
  var flow = []; // step ids of the active flow
  var index = 0;
  var current = null; // { el }: the step on screen
  var countdown = null;
  var transitioning = false; // true while one step leaves and the next mounts
  var calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Animation-matched delays collapse to zero under reduced motion.
  function delay(ms) {
    return calm ? 0 : ms;
  }
  var choice = { name: "", harness: null, models: [], link: null, mint: false };

  // ---- helpers ----------------------------------------------------------
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function svg(viewBox, cls, shapes) {
    var ns = "http://www.w3.org/2000/svg";
    var root = document.createElementNS(ns, "svg");
    root.setAttribute("viewBox", viewBox);
    root.setAttribute("aria-hidden", "true");
    if (cls) root.setAttribute("class", cls);
    shapes.forEach(function (shape) {
      var node = document.createElementNS(ns, shape[0]);
      Object.keys(shape[1]).forEach(function (key) {
        node.setAttribute(key, shape[1][key]);
      });
      root.appendChild(node);
    });
    return root;
  }

  function arrow() {
    var icon = svg("0 0 24 24", null, [["path", { d: "M5 12h14M13 6l6 6-6 6" }]]);
    icon.setAttribute("width", "18");
    icon.setAttribute("height", "18");
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    icon.setAttribute("stroke-width", "2.4");
    icon.setAttribute("stroke-linecap", "round");
    icon.setAttribute("stroke-linejoin", "round");
    return icon;
  }

  function button(label, cls, withArrow) {
    var btn = el("button", "ob-btn" + (cls ? " " + cls : ""));
    btn.type = "button";
    btn.appendChild(el("span", null, label));
    if (withArrow) btn.appendChild(arrow());
    return btn;
  }

  function toast(message) {
    toastEl.textContent = message;
    toastEl.classList.add("is-on");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(function () {
      toastEl.classList.remove("is-on");
    }, 4200);
  }

  function shake(node) {
    node.classList.remove("ob-shake");
    void node.offsetWidth;
    node.classList.add("ob-shake");
  }

  async function api(method, url, body) {
    var resp = await fetch(url, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
      credentials: "same-origin",
    });
    if (resp.status === 401) {
      // /start renders the sign-in page once the session is gone. Guard the
      // reload so a server that keeps answering 401 cannot loop forever.
      if (!sessionStorage.getItem("ob-401")) {
        sessionStorage.setItem("ob-401", "1");
        window.location.href = "/start";
      }
      throw new Error("You're signed out. Please sign in again.");
    }
    sessionStorage.removeItem("ob-401");
    var payload = await resp.json().catch(function () {
      return {};
    });
    if (!resp.ok) {
      var detail = payload && payload.detail;
      if (Array.isArray(detail)) {
        detail = detail
          .map(function (item) {
            return item && item.msg;
          })
          .filter(Boolean)
          .join(". ");
      }
      throw new Error(
        typeof detail === "string" && detail ? detail : "Something went wrong. Try again."
      );
    }
    return payload;
  }

  function firstName() {
    var name = (data.user.name || "").trim().split(/\s+/)[0];
    return name || data.user.email.split("@")[0];
  }

  function harnessById(id) {
    return data.harnesses.find(function (h) {
      return h.id === id;
    });
  }

  // ---- step engine --------------------------------------------------------
  function show(stepEl, dir, onEnter) {
    window.QiipModelInfo.close();
    var previous = current;
    clearInterval(countdown);
    transitioning = true;
    stepEl.style.setProperty("--ob-dir", dir);
    Array.prototype.forEach.call(stepEl.children, function (child, i) {
      child.style.setProperty("--i", i);
    });
    var mount = function () {
      stage.appendChild(stepEl);
      current = { el: stepEl };
      transitioning = false;
      setTimeout(function () {
        if (current.el !== stepEl) return;
        if (onEnter) onEnter();
        else if (stepEl.heading) stepEl.heading.focus({ preventScroll: true });
      }, delay(350));
    };
    var loading = document.getElementById("ob-loading");
    if (loading) loading.remove();
    if (previous && previous.el.isConnected) {
      previous.el.style.setProperty("--ob-dir", dir);
      previous.el.classList.add("is-leaving");
      previous.el.inert = true; // the leaving step can no longer be activated
      setTimeout(function () {
        previous.el.remove();
        mount();
      }, delay(260));
    } else {
      mount();
    }
  }

  function render(dir) {
    var id = flow[index];
    var built = STEPS[id]();
    var step = el("section", "ob-step");
    step.dataset.step = id;
    built.nodes.forEach(function (node) {
      step.appendChild(node);
    });
    backBtn.hidden = !(index > 0 || built.backHome) || id === "done";
    hint.textContent = "";
    if (built.enterHint) {
      hint.append("Press ", el("kbd", null, "Enter ↵"));
    }
    var total = flow.length;
    bar.style.width = total > 1 ? ((index + 1) / total) * 100 + "%" : "0";
    bar.style.opacity = total > 1 ? "1" : "0";
    step.primary = built.primary || null;
    step.heading = step.querySelector(".ob-title");
    if (step.heading) step.heading.tabIndex = -1;
    show(step, dir, built.onEnter); // clears any running countdown first
    if (built.onMount) built.onMount();
  }

  function start(steps) {
    flow = steps;
    index = 0;
    render(1);
  }

  function next() {
    if (transitioning) return;
    if (index < flow.length - 1) {
      index += 1;
      render(1);
    }
  }

  function back() {
    if (transitioning) return;
    if (index > 0) {
      index -= 1;
      render(-1);
    } else {
      goHome(-1);
    }
  }

  function goHome(dir) {
    flow = [data.token ? "home" : "welcome"];
    if (!data.token) flow = ["welcome", "name", "harness", "models", "done"];
    index = 0;
    render(dir || 1);
  }

  backBtn.addEventListener("click", back);
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || event.isComposing || event.repeat) return;
    if (transitioning) return;
    var tag = (event.target.tagName || "").toLowerCase();
    if (tag === "button" || tag === "a") return; // native activation
    var step = current && current.el;
    if (step && step.primary && !step.primary.disabled) {
      event.preventDefault();
      step.primary.click();
    }
  });

  // ---- steps ----------------------------------------------------------------
  var STEPS = {
    welcome: function () {
      var title = el("h1", "ob-title");
      title.append("Welcome to ", el("em", null, "qiip"), ", " + firstName() + ".");
      var go = button("Let's go", null, true);
      go.addEventListener("click", function () {
        choice.mint = true;
        next();
      });
      var actions = el("div", "ob-actions");
      actions.appendChild(go);
      return {
        nodes: [
          el("div", "ob-eyebrow", "Hello"),
          title,
          el(
            "p",
            "ob-sub",
            "Let's connect your coding tool to our models. Three quick questions, about a minute."
          ),
          actions,
        ],
        primary: go,
        enterHint: true,
        onEnter: function () {
          go.focus({ preventScroll: true });
        },
      };
    },

    replace: function () {
      var go = button("Yes, create a new one", "is-danger", true);
      go.addEventListener("click", function () {
        choice.mint = true;
        next();
      });
      var cancel = button("Keep my token", "is-ghost");
      cancel.addEventListener("click", function () {
        goHome(-1);
      });
      var actions = el("div", "ob-actions");
      actions.append(go, cancel);
      return {
        nodes: [
          el("div", "ob-eyebrow", "Heads up"),
          el("h1", "ob-title", "Replace your token?"),
          el(
            "p",
            "ob-sub",
            'Your current token "' +
              data.token.name +
              '" and any older tokens you have will stop working right away. Tools set up with them will need to be set up again.'
          ),
          actions,
        ],
        backHome: true,
      };
    },

    name: function () {
      var input = el("input", "ob-input");
      input.type = "text";
      input.maxLength = 60;
      input.placeholder = "my laptop";
      input.autocomplete = "off";
      input.spellcheck = false;
      input.setAttribute("aria-label", "Token name");
      input.value = choice.name;
      var go = button("Continue", null, true);
      go.disabled = !choice.name.trim();
      input.addEventListener("input", function () {
        choice.name = input.value;
        go.disabled = !input.value.trim();
      });
      go.addEventListener("click", next);
      var chips = el("div", "ob-chips");
      ["my laptop", "work desktop", "dev server"].forEach(function (label) {
        var chip = el("button", "ob-chip", label);
        chip.type = "button";
        chip.addEventListener("click", function () {
          input.value = label;
          input.dispatchEvent(new Event("input"));
          input.focus();
        });
        chips.appendChild(chip);
      });
      var actions = el("div", "ob-actions");
      actions.appendChild(go);
      return {
        nodes: [
          el("div", "ob-eyebrow", "Question 1 of 3"),
          el("h1", "ob-title", "Give your token a name."),
          el(
            "p",
            "ob-sub",
            "A token is your personal key to qiip. Name it after the computer you'll use it on."
          ),
          input,
          chips,
          actions,
        ],
        primary: go,
        enterHint: true,
        onEnter: function () {
          input.focus({ preventScroll: true });
        },
      };
    },

    harness: function () {
      var grid = el("div", "ob-choices");
      grid.setAttribute("role", "group");
      grid.setAttribute("aria-label", "Coding tools");
      data.harnesses.forEach(function (harness) {
        var card = el("button", "ob-choice");
        card.type = "button";
        card.disabled = !harness.available;
        card.setAttribute("aria-pressed", String(choice.harness === harness.id));
        card.append(
          el("span", "ob-choice-label", harness.label),
          el("span", "ob-choice-note", harness.tagline)
        );
        if (!harness.available) card.appendChild(el("span", "ob-soon", "Soon"));
        card.addEventListener("click", function () {
          if (choice.harness !== harness.id) choice.models = [];
          choice.harness = harness.id;
          Array.prototype.forEach.call(grid.children, function (other) {
            other.setAttribute("aria-pressed", String(other === card));
          });
          if (grid.dataset.picked) return; // one advance per visit
          grid.dataset.picked = "1";
          setTimeout(next, delay(280)); // let the check land, then move on
        });
        grid.appendChild(card);
      });
      return {
        nodes: [
          el("div", "ob-eyebrow", choice.mint ? "Question 2 of 3" : "Step 1 of 2"),
          el("h1", "ob-title", "Which tool do you code with?"),
          el("p", "ob-sub", "Pick one. You can come back and set up another any time."),
          grid,
        ],
        backHome: !choice.mint,
      };
    },

    models: function () {
      var harness = harnessById(choice.harness) || data.harnesses[0];
      choice.harness = harness.id;
      var multi = harness.multi_model;
      var available = data.models;
      if (!multi && choice.models.length > 1) choice.models = choice.models.slice(0, 1);
      if (!choice.models.length && multi && data.token && data.token.models && !choice.mint) {
        choice.models = data.token.models.filter(function (m) {
          return available.indexOf(m) !== -1;
        });
      }
      if (!choice.models.length && available.length === 1) choice.models = [available[0]];

      var go = button(choice.mint ? "Create my token" : "Get my setup command", null, true);
      var grid = el("div", "ob-choices is-models");
      grid.setAttribute("role", "group");
      grid.setAttribute("aria-label", "Models");
      var sync = function () {
        go.disabled = choice.models.length === 0;
        Array.prototype.forEach.call(grid.querySelectorAll("button[data-model]"), function (card) {
          card.setAttribute(
            "aria-pressed",
            String(choice.models.indexOf(card.dataset.model) !== -1)
          );
        });
      };
      available.forEach(function (model) {
        var card = el("button", "ob-choice is-mono" + (multi ? " is-multi" : ""));
        card.type = "button";
        card.dataset.model = model;
        card.appendChild(el("span", "ob-choice-label", window.QiipModelInfo.displayName(model)));
        card.addEventListener("click", function () {
          var at = choice.models.indexOf(model);
          if (!multi) choice.models = [model];
          else if (at === -1) choice.models.push(model);
          else choice.models.splice(at, 1);
          sync();
        });
        grid.appendChild(window.QiipModelInfo.wrap(card, model, (data.model_details || {})[model]));
      });
      sync();

      go.addEventListener("click", async function () {
        if (go.disabled) return;
        go.disabled = true; // blocks Enter and programmatic clicks too
        go.classList.add("is-busy");
        try {
          if (choice.mint) {
            var minted = await api("POST", "/onboarding/token", {
              name: choice.name.trim(),
              models: choice.models,
            });
            data.token = minted.token;
            choice.mint = false; // a retry must not mint (and revoke) again
          }
          choice.link = await api("POST", "/onboarding/setup-link", {
            harness: choice.harness,
            models: choice.models,
          });
          next();
        } catch (err) {
          go.classList.remove("is-busy");
          go.disabled = false;
          shake(go);
          toast(err.message);
        }
      });

      var nodes = [
        el("div", "ob-eyebrow", choice.mint ? "Question 3 of 3" : "Step 2 of 2"),
        el("h1", "ob-title", multi ? "Which models do you want?" : "Which model do you want?"),
      ];
      if (!available.length) {
        var retry = button("Check again", "is-ghost");
        retry.addEventListener("click", async function () {
          retry.classList.add("is-busy");
          try {
            data = await api("GET", "/onboarding/state");
            render(1);
          } catch (err) {
            retry.classList.remove("is-busy");
            toast(err.message);
          }
        });
        var retryRow = el("div", "ob-actions");
        retryRow.appendChild(retry);
        nodes.push(
          el(
            "p",
            "ob-sub",
            "No models are online right now. They come and go as capacity changes, so check back in a few minutes."
          ),
          retryRow
        );
        return { nodes: nodes };
      }
      var actions = el("div", "ob-actions");
      actions.appendChild(go);
      nodes.push(
        el(
          "p",
          "ob-sub",
          multi
            ? harness.label + " can switch between models. Choose as many as you like."
            : harness.label + " works with one model at a time. You can change it later."
        ),
        grid,
        actions
      );
      return { nodes: nodes, primary: go, enterHint: true, onEnter: function () {
        window.QiipModelInfo.sizeChoices(grid);
        if (document.fonts) document.fonts.ready.then(function () {
          if (grid.isConnected) window.QiipModelInfo.sizeChoices(grid);
        });
      } };
    },

    done: function () {
      if (!choice.link) return STEPS.home();
      var harness = harnessById(choice.link.harness);
      var check = svg("0 0 56 56", "ob-check", [
        ["circle", { cx: "28", cy: "28", r: "25" }],
        ["path", { d: "M17 29l8 8 15-17" }],
      ]);

      var box = el("div", "ob-command");
      var code = el("code", null, choice.link.command);
      var copy = el("button", "ob-copy", "Copy");
      copy.type = "button";
      box.append(code, copy);
      copy.addEventListener("click", async function () {
        var copied = false;
        try {
          await navigator.clipboard.writeText(choice.link.command);
          copied = true;
        } catch (err) {
          var range = document.createRange();
          range.selectNodeContents(code);
          var selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          try {
            copied = document.execCommand("copy");
          } catch (ignored) {
            copied = false;
          }
        }
        if (!copied) {
          toast("Couldn't copy for you. The line is selected: press Ctrl+C.");
          return;
        }
        copy.textContent = "Copied!";
        copy.classList.add("is-done");
        setTimeout(function () {
          copy.textContent = "Copy";
          copy.classList.remove("is-done");
        }, 2200);
      });

      var meta = el("div", "ob-meta");
      meta.setAttribute("aria-live", "off"); // a ticking clock is not news
      var timer = el("strong");
      var expiry = el("span");
      expiry.append("Link works for ", timer);
      var then = el("span");
      then.append("Then start it by typing ", el("code", null, choice.link.run_command));
      meta.append(expiry, then);

      var finish = button("I'm done", null, true);
      finish.addEventListener("click", async function () {
        finish.classList.add("is-busy");
        try {
          data = await api("GET", "/onboarding/state");
        } catch (err) {
          /* fall through with the state we have */
        }
        choice = { name: "", harness: null, models: [], link: null, mint: false };
        goHome(1);
      });
      var renew = button("Get a new link", "is-ghost");
      renew.hidden = true;
      renew.addEventListener("click", async function () {
        renew.classList.add("is-busy");
        try {
          choice.link = await api("POST", "/onboarding/setup-link", {
            harness: choice.link.harness,
            models: choice.models,
          });
          render(1);
        } catch (err) {
          renew.classList.remove("is-busy");
          toast(err.message);
        }
      });
      var actions = el("div", "ob-actions");
      actions.append(finish, renew);

      var deadline = new Date(choice.link.expires_at).getTime();
      var tick = function () {
        var left = Math.max(0, Math.round((deadline - Date.now()) / 1000));
        timer.textContent =
          Math.floor(left / 60) + ":" + String(left % 60).padStart(2, "0");
        if (left === 0) {
          clearInterval(countdown);
          expiry.textContent = "This link has expired.";
          toast("This link has expired. Get a new one below.");
          box.classList.add("is-expired");
          copy.disabled = true;
          renew.hidden = false;
        }
      };

      return {
        nodes: [
          check,
          el("h1", "ob-title", "You're all set."),
          el(
            "p",
            "ob-sub",
            "Open a terminal, paste this line, and press Enter. It sets up " +
              harness.label +
              " for you."
          ),
          box,
          meta,
          actions,
        ],
        onEnter: function () {
          copy.focus({ preventScroll: true });
        },
        onMount: function () {
          tick();
          countdown = setInterval(tick, 1000);
        },
      };
    },

    home: function () {
      var token = data.token;
      var title = el("h1", "ob-title");
      title.append("Hi ", el("em", null, firstName()), "!");

      var card = el("div", "ob-card");
      var row = el("div", "ob-token-row");
      row.append(
        el("span", "ob-token-name", token.name),
        el("span", "ob-token-key", token.prefix + "…")
      );
      var made = new Date(token.created_at);
      var facts = el("div", "ob-meta");
      facts.appendChild(
        el(
          "span",
          null,
          "Created " +
            made.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
        )
      );
      facts.appendChild(
        el(
          "span",
          null,
          token.last_used_at
            ? "Last used " +
                new Date(token.last_used_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                })
            : "Not used yet"
        )
      );
      var labelRow = el("div", "ob-token-row");
      var saved = el("span", "ob-saved", "Saved");
      labelRow.append(
        el(
          "span",
          "ob-label",
          token.models ? "Models this token can use" : "This token can use every model"
        ),
        saved
      );

      var active = token.models ? token.models.slice() : data.models.slice();
      var all = data.models.slice();
      active.forEach(function (model) {
        if (all.indexOf(model) === -1) all.push(model);
      });
      var chips = el("div", "ob-models");
      var saving = Promise.resolve();
      // Only a personal token's scope is editable. An older token is
      // unrestricted, and narrowing it here could not be undone.
      var editable = token.exportable;
      var paint = function () {
        Array.prototype.forEach.call(chips.querySelectorAll("button[data-model]"), function (chip) {
          chip.setAttribute("aria-pressed", String(active.indexOf(chip.dataset.model) !== -1));
        });
      };
      all.forEach(function (model) {
        var offline = data.models.indexOf(model) === -1;
        var name = window.QiipModelInfo.displayName(model);
        var chip = el("button", "ob-model", offline ? name + " (offline)" : name);
        chip.type = "button";
        chip.dataset.model = model;
        chip.disabled = !editable;
        if (offline) chip.classList.add("is-offline");
        chip.addEventListener("click", function () {
          var at = active.indexOf(model);
          if (at !== -1 && active.length === 1) {
            shake(chip);
            toast("Your token needs at least one model.");
            return;
          }
          if (at === -1) active.push(model);
          else active.splice(at, 1);
          paint();
          var snapshot = active.slice();
          saving = saving.then(async function () {
            try {
              var result = await api("PUT", "/onboarding/token/models", { models: snapshot });
              data.token = result.token;
              saved.classList.add("is-on");
              clearTimeout(saved.timer);
              saved.timer = setTimeout(function () {
                saved.classList.remove("is-on");
              }, 1600);
            } catch (err) {
              // Put the chips back to what the server actually holds.
              active = (data.token.models || []).slice();
              paint();
              toast(err.message);
            }
          });
        });
        chips.appendChild(window.QiipModelInfo.wrap(chip, model, (data.model_details || {})[model]));
      });
      paint();
      card.append(row, facts, labelRow, chips);
      if (!all.length) {
        card.appendChild(el("p", "ob-sub", "No models are online right now."));
      }

      var setup = button("Set up a tool", null, true);
      setup.disabled = !token.exportable;
      setup.addEventListener("click", function () {
        choice = { name: "", harness: null, models: [], link: null, mint: false };
        start(["harness", "models", "done"]);
      });
      var fresh = button("Create a new token", "is-ghost");
      fresh.addEventListener("click", function () {
        choice = { name: "", harness: null, models: [], link: null, mint: false };
        start(["replace", "name", "harness", "models", "done"]);
      });
      var actions = el("div", "ob-actions");
      actions.append(setup, fresh);

      var nodes = [el("div", "ob-eyebrow", "Your token"), title, card, actions];
      if (!token.exportable) {
        nodes.push(
          el(
            "p",
            "ob-sub",
            "This token was made before the new setup flow. Create a new token to set up a tool in one step."
          )
        );
      }
      return { nodes: nodes };
    },
  };

  // ---- boot ---------------------------------------------------------------------
  document.getElementById("ob-theme").addEventListener("click", function () {
    var theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch (err) {
      /* private mode */
    }
  });

  api("GET", "/onboarding/state")
    .then(function (payload) {
      data = payload;
      document.getElementById("ob-who").textContent = data.user.email;
      goHome(1);
    })
    .catch(function (err) {
      var loading = document.getElementById("ob-loading");
      if (loading) loading.remove();
      var retry = button("Try again", "is-ghost");
      retry.addEventListener("click", function () {
        window.location.reload();
      });
      var failed = el("section", "ob-step");
      failed.append(
        el("h1", "ob-title", "We couldn't load your page."),
        el("p", "ob-sub", err.message),
        retry
      );
      show(failed, 1);
    });
})();
