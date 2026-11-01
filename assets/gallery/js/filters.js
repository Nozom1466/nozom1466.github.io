export const filterKeys = ["subject", "location", "color", "year"];

export function matchesFilters(metadata, filters) {
  return filterKeys.every(key => {
    const selected = filters[key] || "";
    const values = key === "location" ? metadata.locations : [metadata[key]].filter(Boolean);
    return !selected || (selected === "unknown" ? values.length === 0 : values.includes(selected));
  });
}

export function setupFilters(items, onChange) {
  const form = document.getElementById("gallery-filters");
  if (!form) return;
  const controls = filterKeys.map(key => form.elements.namedItem(key));
  const metadata = new Map(Array.from(items, item => [item, {
    subject: item.dataset.subject || "",
    locations: JSON.parse(item.dataset.locations || "[]"),
    color: item.dataset.color || "",
    year: item.dataset.year || ""
  }]));
  function readURL() {
    const query = new URLSearchParams(location.search);
    controls.forEach(control => {
      const value = query.get(control.name) || "";
      control.value = Array.from(control.options).some(option => option.value === value) ? value : "";
    });
  }
  function apply() {
    const filters = Object.fromEntries(controls.map(control => [control.name, control.value]));
    let count = 0;
    items.forEach(item => {
      const visible = matchesFilters(metadata.get(item), filters);
      item.classList.toggle("tag-hidden", !visible);
      item.style.display = visible ? "" : "none";
      if (visible) count++;
    });
    document.getElementById("gallery-empty").hidden = count !== 0;
  }
  function updateURL() {
    const url = new URL(location.href);
    url.searchParams.delete("tag");
    controls.forEach(control => {
      if (control.value) url.searchParams.set(control.name, control.value);
      else url.searchParams.delete(control.name);
    });
    url.hash = "";
    history.pushState(null, "", url);
    apply();
    onChange();
  }
  form.addEventListener("submit", event => event.preventDefault());
  form.addEventListener("change", updateURL);
  form.addEventListener("reset", event => {
    event.preventDefault();
    controls.forEach(control => { control.value = ""; });
    updateURL();
  });
  window.addEventListener("popstate", () => { readURL(); apply(); onChange(); });
  readURL();
  apply();
}
