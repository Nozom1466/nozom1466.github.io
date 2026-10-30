// Native selects remain the source of truth for filtering and URL restoration.
// Enhance only after the gallery has initialized.
export function enhanceDropdowns(form) {
  if (!form) return;
  const widgets = [];
  const swatches = { red: "#b95853", orange: "#c78e55", yellow: "#c4ad5f", green: "#718b71", blue: "#6e8ba9", black: "#333333", white: "#fafafa" };
  function closeAll(except) {
    widgets.forEach(widget => { if (widget !== except) widget.close(); });
  }
  form.querySelectorAll("select").forEach(select => {
    const field = select.parentElement;
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "filter-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    const title = select.getAttribute("aria-label");
    const text = document.createElement("span");
    text.className = "filter-trigger-text";
    const arrow = document.createElement("span");
    arrow.className = "filter-chevron";
    arrow.setAttribute("aria-hidden", "true");
    trigger.append(text, arrow);
    const menu = document.createElement("div");
    menu.id = "filter-menu-" + select.name;
    menu.className = "filter-menu";
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", title);
    menu.hidden = true;
    trigger.setAttribute("aria-controls", menu.id);
    const options = Array.from(select.options).map(option => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-option";
      button.tabIndex = -1;
      button.setAttribute("role", "option");
      button.dataset.value = option.value;
      if (!option.value) button.classList.add("filter-option-clear");
      if (select.name === "color" && swatches[option.value]) {
        const swatch = document.createElement("span");
        swatch.className = "filter-swatch";
        swatch.style.backgroundColor = swatches[option.value];
        swatch.setAttribute("aria-hidden", "true");
        button.append(swatch);
      }
      const label = document.createElement("span");
      label.className = "filter-option-text";
      const parts = option.textContent.split(" · ");
      label.textContent = parts[0];
      if (parts[1]) {
        const subtitle = document.createElement("small");
        subtitle.textContent = parts[1];
        label.append(subtitle);
      }
      const check = document.createElement("span");
      check.className = "filter-check";
      check.setAttribute("aria-hidden", "true");
      check.textContent = "✓";
      button.append(label, check);
      button.addEventListener("click", () => {
        select.value = option.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        widget.sync();
        widget.close(true);
      });
      menu.append(button);
      return button;
    });
    const widget = {
      close(focus = false) {
        menu.hidden = true;
        trigger.setAttribute("aria-expanded", "false");
        field.classList.remove("is-open");
        if (focus) trigger.focus();
      },
      open(index = select.selectedIndex) {
        closeAll(widget);
        menu.hidden = false;
        trigger.setAttribute("aria-expanded", "true");
        field.classList.add("is-open");
        options[Math.max(0, index)].focus();
      },
      sync() {
        const option = select.options[select.selectedIndex] || select.options[0];
        text.textContent = option.textContent.split(" · ")[0];
        trigger.setAttribute("aria-label", title + "：" + option.textContent);
        trigger.classList.toggle("has-value", Boolean(select.value));
        options.forEach(button => button.setAttribute("aria-selected", String(button.dataset.value === select.value)));
      }
    };
    widgets.push(widget);
    let typed = "", lastTyped = 0;
    field.addEventListener("keydown", event => {
      const index = options.indexOf(document.activeElement);
      if (event.key === "Escape") {
        if (!menu.hidden) { event.preventDefault(); event.stopPropagation(); widget.close(true); }
      } else if (event.key === "Tab") {
        widget.close();
      } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        if (menu.hidden) widget.open(event.key === "End" ? options.length - 1 : select.selectedIndex);
        else {
          const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1 : (index + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
          options[next].focus();
        }
      } else if (!menu.hidden && event.key.length === 1 && event.key !== " " && !event.ctrlKey && !event.metaKey) {
        typed = Date.now() - lastTyped > 700 ? event.key : typed + event.key;
        lastTyped = Date.now();
        const match = Array.from(select.options).findIndex(option => option.textContent.toLowerCase().split(" · ").some(part => part.startsWith(typed.toLowerCase())));
        if (match >= 0) { event.preventDefault(); options[match].focus(); }
      }
    });
    trigger.addEventListener("click", () => menu.hidden ? widget.open() : widget.close());
    field.addEventListener("focusout", event => {
      if (!field.contains(event.relatedTarget)) widget.close();
    });
    field.append(trigger, menu);
    select.hidden = true;
    widget.sync();
  });
  document.addEventListener("pointerdown", event => {
    if (!event.target.closest(".gallery-filter-field")) closeAll();
  });
  const sync = () => widgets.forEach(widget => widget.sync());
  form.addEventListener("change", sync);
  form.addEventListener("reset", () => { closeAll(); queueMicrotask(sync); });
  window.addEventListener("popstate", () => { closeAll(); queueMicrotask(sync); });
}

enhanceDropdowns(document.getElementById("gallery-filters"));
