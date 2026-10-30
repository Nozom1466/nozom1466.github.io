import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const code = readFileSync(new URL("../assets/gallery/js/filters.js", import.meta.url), "utf8");
const { matchesFilters, setupFilters } = await import("data:text/javascript;base64," + Buffer.from(code).toString("base64"));
const photo = { subject: "architecture", locations: ["上海", "成都"], color: "red", year: "2025" };
assert(matchesFilters(photo, {}));
assert(matchesFilters(photo, { subject: "architecture", location: "成都", color: "red", year: "2025" }));
assert(!matchesFilters(photo, { location: "匹兹堡" }));
assert(!matchesFilters(photo, { subject: "animal" }));
assert(!matchesFilters(photo, { year: "2024" }));
assert(!matchesFilters(photo, { color: "unknown" }));
assert(matchesFilters({ ...photo, locations: [], color: "", year: "" }, { location: "unknown", color: "unknown", year: "unknown" }));

const controls = Object.entries({
  subject: ["", "architecture", "animal"], location: ["", "上海", "成都", "unknown"],
  color: ["", "red", "blue", "unknown"], year: ["", "2025", "unknown"]
}).map(([name, values]) => ({ name, value: "", options: values.map(value => ({value})) }));
const handlers = {};
const form = { elements: { namedItem: name => controls.find(c => c.name === name) }, addEventListener: (name, fn) => { handlers[name] = fn; } };
const count = {}, empty = {}, events = {};
globalThis.document = { getElementById: id => ({ "gallery-filters": form, "gallery-count": count, "gallery-empty": empty })[id] };
globalThis.location = new URL("https://example.test/gallery/?subject=architecture&location=成都");
globalThis.history = { pushState: (_, __, url) => { globalThis.location = new URL(url); } };
globalThis.window = { addEventListener: (name, fn) => { events[name] = fn; } };
const item = { dataset: { ...photo, locations: JSON.stringify(photo.locations) }, style: {}, classList: { toggle: (_, hidden) => { item.hidden = hidden; } } };
let layouts = 0;
setupFilters([item], () => layouts++);
assert(!item.hidden);
assert.equal(count.textContent, "1 / 1 张");
controls.find(c => c.name === "subject").value = "animal";
handlers.change();
assert(item.hidden);
assert.equal(empty.hidden, false);
assert.equal(location.searchParams.get("subject"), "animal");
handlers.reset({ preventDefault() {} });
assert(!item.hidden);
assert.equal(location.search, "");
globalThis.location = new URL("https://example.test/gallery/?year=1900&color=invalid");
events.popstate();
assert(!item.hidden);
assert.equal(layouts, 3);

const html = readFileSync(new URL("../.gallery-preview/gallery/index.html", import.meta.url), "utf8");
assert.equal((html.match(/class=["']?gallery-item[\s"'>]/g) || []).length, 22);
for (const key of ["subject", "location", "color", "year"]) assert(new RegExp('name=["\x27]?' + key + '[\\s"\x27>]').test(html));
assert.equal((html.match(/data-subject=/g) || []).length, 22);
assert.equal((html.match(/data-locations=/g) || []).length, 22);
assert(!html.includes('class="tag-link"'));
console.log("Gallery tests passed: filtering, multiple cities, missing fields, URL restore, reset, empty state, 22 rendered photos.");
