// SnapNote snapshot adapter. Never treat note contents as trusted HTML.
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function safeLink(value) {
  try { const u = new URL(value); return ['http:', 'https:'].includes(u.protocol) && !u.username && !u.password ? u.href : null; } catch { return null; }
}
export function mediaPath(name) {
  return typeof name === 'string' && /^[\w.-]+\.(jpg|jpeg|png|webp|gif|mp4|webm|mp3|m4a|ogg|wav)$/i.test(name) && !name.includes('..') ? `/api/moments/media/${encodeURIComponent(name)}` : null;
}
export function parseLines(content) {
  const ops = typeof content === 'string' ? JSON.parse(content) : content;
  if (!Array.isArray(ops)) throw new Error('笔记格式不是操作列表');
  const lines = []; let parts = [];
  for (const op of ops) {
    if (!op || !('insert' in op)) continue;
    const attrs = op.attributes || {};
    if (typeof op.insert !== 'string') { parts.push({embed:op.insert,attrs}); continue; }
    const chunks = op.insert.split('\n');
    chunks.forEach((text, i) => {
      if (text) parts.push({text,attrs});
      if (i < chunks.length - 1) { lines.push({parts,attrs}); parts = []; }
    });
  }
  if (parts.length) lines.push({parts,attrs:{}});
  return lines;
}
// Count Unicode graphemes; a Chinese character or an emoji counts as one.
export function excerpt(content, limit = 200) {
  const ops = typeof content === 'string' ? JSON.parse(content) : content;
  if (!Array.isArray(ops)) throw new Error('笔记格式不是操作列表');
  const segment = text => [...new Intl.Segmenter('zh-CN',{granularity:'grapheme'}).segment(text)].map(x=>x.segment);
  const count = ops.reduce((n,op)=>n+(typeof op.insert==='string'?segment(op.insert).filter(c=>!/^\s+$/.test(c)).length:0),0);
  if(count <= limit) return {content:ops,truncated:false,count};
  let used=0, done=false; const result=[]; const trailingImages=[];
  for(const op of ops){
    if(typeof op.insert !== 'string') {
      if(!done) result.push(op); else if(op.insert?.image) trailingImages.push(op);
      continue;
    }
    if(done) continue;
    let text='';
    for(const c of segment(op.insert)) {
      if(!/^\s+$/.test(c)) { if(used >= limit){done=true;break;} used++; }
      text+=c;
    }
    if(text) result.push({...op,insert:text});
  }
  result.push({insert:'…\n'});
  for(const image of trailingImages) result.push(image,{insert:'\n'});
  return {content:result,truncated:true,count};
}
function inline(part, media=null) {
  const a = part.attrs;
  if (part.embed) {
    const e = part.embed;
    if (e.image) {
      const published=media?.[e.image];
      const safeAsset=url=>typeof url==='string'&&!url.includes('..')&&(/^\/(?!\/)[\w/.-]+$/.test(url)||/^https:\/\/snapnote-test-1316371247\.cos\.ap-shanghai\.myqcloud\.com\/SnapNote_Sync\/resources\/[\w.-]+\.(jpe?g|png|webp|gif)$/i.test(url));
      const src=media===null?mediaPath(e.image):safeAsset(published?.original)?published.original:null;
      const thumbnail=media===null?src+'?thumbnail=1':safeAsset(published?.thumbnail)?published.thumbnail:null;
      return src&&thumbnail ? `<button class="photo" data-image="${escapeHTML(src)}" aria-label="放大图片"><img src="${escapeHTML(thumbnail)}" alt="笔记图片" loading="lazy" decoding="async"><span class="photo-zoom" aria-hidden="true"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5M10.5 7.5v6M7.5 10.5h6"/></svg></span></button>` : '<span class="unsupported">图片路径不受支持</span>';
    }
    if (e.formula) return `<span class="formula" data-formula="${escapeHTML(e.formula)}">${escapeHTML(e.formula)}</span>`;
    return `<span class="unsupported">暂不支持的内容：${escapeHTML(Object.keys(e).join(', '))}</span>`;
  }
  let html = escapeHTML(part.text);
  if (a.code) html = `<code>${html}</code>`;
  if (a.bold) html = `<strong>${html}</strong>`;
  if (a.italic) html = `<em>${html}</em>`;
  if (a.strike) html = `<s>${html}</s>`;
  if (a.underline) html = `<u>${html}</u>`;
  const url = safeLink(a.link);
  return url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${html}</a>` : html;
}
export function renderContent(content, media=null) {
  const lines = parseLines(content); let html = ''; let table = null; let row = null; let list = null;
  const closeTable = () => { if (table) { html += '</tr></tbody></table></div>'; table = row = null; } };
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  let photos = [], photoBlanks = 0;
  const closePhotos = () => {
    if (!photos.length) return;
    html += `<div class="photo-grid" data-columns="${Math.min(3,photos.length)}" role="group" aria-label="图片组">${photos.map(part=>inline(part,media)).join('')}</div>`;
    photos = []; photoBlanks = 0;
  };
  // Split ordinary mixed text/image lines without moving images past captions.
  const displayLines = lines.flatMap(line => {
    if(Object.keys(line.attrs).length || !line.parts.some(p=>p.embed?.image)) return [line];
    const chunks=[]; let parts=[];
    for(const part of line.parts){
      if(part.embed?.image){if(parts.length)chunks.push({parts,attrs:{}});parts=[];chunks.push({parts:[part],attrs:{}});}
      else parts.push(part);
    }
    if(parts.length)chunks.push({parts,attrs:{}});
    return chunks;
  });
  for (const line of displayLines) {
    const meaningful=line.parts.filter(p=>p.embed || p.text?.trim());
    const photoLine=!Object.keys(line.attrs).length && meaningful.length && meaningful.every(p=>p.embed?.image && mediaPath(p.embed.image));
    if(photoLine){closeTable();closeList();photos.push(...meaningful);continue;}
    if(photos.length && !meaningful.length && !Object.keys(line.attrs).length){photoBlanks++;continue;}
    const blankCount=photoBlanks;closePhotos();
    if(blankCount)html+='<div class="line blank"><br></div>';
    const a = line.attrs, value = line.parts.map(part=>inline(part,media)).join('');
    if (a.table) {
      closeList();
      if (table !== a.table) { closeTable(); html += '<div class="table-scroll"><table aria-label="笔记表格"><tbody><tr>'; table = a.table; row = a['table-row']; }
      if (row !== a['table-row']) { html += '</tr><tr>'; row = a['table-row']; }
      html += `<td>${value || '&nbsp;'}</td>`; continue;
    }
    closeTable();
    if (a.list === 'ordered' || a.list === 'bullet') {
      const kind = a.list === 'ordered' ? 'ol' : 'ul';
      if (list !== kind) { closeList(); html += `<${kind}>`; list = kind; }
      html += `<li>${value}</li>`; continue;
    }
    closeList();
    if (a.list === 'checked' || a.list === 'unchecked') html += `<div class="todo"><input type="checkbox" aria-label="笔记待办（只读）" disabled ${a.list === 'checked' ? 'checked' : ''}><span>${value}</span></div>`;
    else if (a.callout) html += `<aside class="callout"><span aria-hidden="true">${escapeHTML(a['callout-emoji'] || '›')}</span><div>${value}</div></aside>`;
    else if (a.blockquote) html += `<blockquote>${value}</blockquote>`;
    else if (Number.isInteger(a.header) && a.header >= 1 && a.header <= 6) html += `<h${a.header}>${value}</h${a.header}>`;
    else html += `<div class="line${value ? '' : ' blank'}">${value || '<br>'}</div>`;
  }
  closePhotos(); closeTable(); closeList(); return html;
}
export function visibleNotes(records) {
  const latest = new Map();
  for (const record of records) {
    const note = record?.note;
    if (!note?.uuid) continue;
    const prev = latest.get(note.uuid);
    const time = Number(record.stateVector?.lastModifiedMs ?? note.updatedAtMs ?? 0);
    if (!prev || time >= prev.time) latest.set(note.uuid,{record,time});
  }
  return [...latest.values()].map(x=>x.record).filter(r=>!r.note.hasMarkDeleted && !r.stateVector?.deleted && !r.note.isLocked && !r.note.isArchived).sort((a,b)=>Number(b.note.createdAtMs)-Number(a.note.createdAtMs));
}
