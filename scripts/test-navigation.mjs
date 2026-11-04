import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const destination = process.argv[2];
assert(destination, 'Pass the Hugo build output directory');
// Hugo's HTML minifier may remove optional attribute quotes.
const read = route => readFileSync(join(destination, route, 'index.html'), 'utf8')
  .replace(/\b(class|id|href|aria-current|aria-expanded)=([^\s"'<>`=]+)/g, '$1="$2"');
const home = read('');
const mainNav = home.match(/<nav\b[^>]*class="site-nav"[^>]*>([\s\S]*?)<\/nav>/)?.[1];
assert(mainNav, 'Home navigation exists');
const links = html => [...html.matchAll(/<a\b[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)]
  .map(([, href, content]) => ({ href, text: content.replace(/<[^>]*>/g, '').trim() }));
const expected = links(mainNav);
assert.deepEqual(expected.map(link => link.text), ['首页', '归档', '标签', '展廊', '书馆', '此刻']);

for (const route of ['', 'gallery', 'library', 'moments']) {
  const html = read(route);
  assert(!html.includes('menu-item-search'), `${route || 'home'}: no search navigation`);
  assert(!html.includes('class="popup-trigger"'), `${route || 'home'}: no search trigger`);
  if (!['gallery', 'library'].includes(route)) continue;
  const nav = html.match(new RegExp(`<nav\\b[^>]*id="${route}-navigation"[^>]*>([\\s\\S]*?)<\\/nav>`))?.[1];
  assert(nav, `${route}: section navigation exists`);
  assert.deepEqual(links(nav), expected, `${route}: matches home navigation order, URLs and labels`);
  assert.equal((nav.match(/aria-current="page"/g) || []).length, 1, `${route}: one active link`);
  assert(nav.includes(`href="/${route}/" aria-current="page"`), `${route}: correct active link`);
  assert(nav.includes('moments-nav-icon'), `${route}: same Moments icon`);
  assert(nav.includes('id="toggle-theme"'), `${route}: theme switch retained`);
  assert(html.includes('aria-expanded="false"'), `${route}: accessible collapsed mobile menu`);
  assert(/@media\s*\(max-width:\s*1023px\)/.test(html), `${route}: collapse before desktop navigation overflows`);
}
console.log('Navigation tests passed: identical six links, active states, Moments icon, theme controls, responsive menus, no search.');
