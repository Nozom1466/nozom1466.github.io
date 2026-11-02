import {escapeHTML, safeLink} from '../../static/moments/render.mjs';
import {COS_ORIGIN} from '../../static/moments/cos-feed.mjs';

const extensions = {audio: /\.(mp3|m4a|ogg|wav)$/i, image: /\.(jpe?g|png|webp|gif)$/i};
const plain = line => line?.parts.map(p => p.text || '').join('').trim() || '';
const blank = line => line && !plain(line) && !line.parts.some(p => p.embed);
const escape = value => escapeHTML(value).replaceAll('{', '&#123;').replaceAll('}', '&#125;');

export function resourceURL(value, kind) {
  if (typeof value !== 'string') throw Error('音乐附件格式不支持，请使用音频附件或 COS 直链');
  const ext = extensions[kind];
  if (/^[\w.-]+$/.test(value) && !value.includes('..') && ext.test(value)) {
    return COS_ORIGIN + '/SnapNote_Sync/resources/' + encodeURIComponent(value);
  }
  const link = safeLink(value);
  if (link) {
    const u = new URL(link);
    if (u.protocol === 'https:' && !u.search && !u.hash && ext.test(u.pathname) &&
      ((u.origin === COS_ORIGIN && u.pathname.startsWith('/SnapNote_Sync/resources/')) ||
       (u.hostname === 'myblog-1316371247.cos.ap-shanghai.myqcloud.com' && u.pathname.startsWith('/myblog/')))) return link;
  }
  throw Error(kind === 'audio' ? '音乐需要 MP3/M4A/OGG/WAV 附件或现有 COS 音频直链' : 'audio_cover 需要图片附件或现有 COS 图片直链');
}

function audioSource(line) {
  const parts = line?.parts.filter(p => p.embed || p.text?.trim()) || [];
  if (parts.length !== 1) throw Error('audio 后需要单独一行音频附件或 COS 音频直链');
  const p = parts[0];
  if (!p.embed) return resourceURL(p.attrs?.link || p.text.trim(), 'audio');
  const value = p.embed.audio ?? p.embed.file ?? p.embed.attachment;
  // Accept explicit resource paths only, never guess from an attachment's display name.
  return resourceURL(typeof value === 'object' && value ? (value.url ?? value.path ?? value.src) : value, 'audio');
}

// The block is deliberately ordered: audio, optional artist, attachment, optional audio_cover.
export function parseAudioBlock(lines, start) {
  const line = lines[start];
  if (!line || line.parts.some(p => p.embed) || Object.keys(line.attrs).length) return null;
  const match = plain(line).match(/^audio\s*[：:]\s*(.*)$/i);
  if (!match) return null;
  const name = match[1].trim();
  if (!name) throw Error('audio 需要填写曲名');
  let cursor = start + 1, artist = '', cover = '';
  const skip = () => { while (blank(lines[cursor])) cursor++; };
  skip();
  const artistMatch = !lines[cursor]?.parts.some(p => p.embed) && plain(lines[cursor]).match(/^artist\s*[：:]\s*(.*)$/i);
  if (artistMatch) { artist = artistMatch[1].trim(); cursor++; skip(); }
  const url = audioSource(lines[cursor]); cursor++;
  let end = cursor - 1;
  skip();
  const coverMatch = !lines[cursor]?.parts.some(p => p.embed) && plain(lines[cursor]).match(/^audio_cover\s*[：:]\s*(.*)$/i);
  if (coverMatch) {
    if (coverMatch[1].trim()) cover = resourceURL(coverMatch[1].trim(), 'image');
    else {
      cursor++; skip();
      const parts = lines[cursor]?.parts.filter(p => p.embed || p.text?.trim()) || [];
      if (parts.length !== 1 || !parts[0].embed?.image) throw Error('audio_cover 下一行需要插入封面图片');
      cover = resourceURL(parts[0].embed.image, 'image');
    }
    end = cursor;
  }
  const attrs = Object.entries({url, name, artist, cover, theme: '#448aff', loop: 'none'})
    .map(([key, value]) => `data-${key}="${escape(value)}"`).join(' ');
  return {end, url, cover, html: `<music-placeholder ${attrs}></music-placeholder>`};
}
