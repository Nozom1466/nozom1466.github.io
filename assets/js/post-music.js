// Blog and Review use the same music-placeholder markup and APlayer version.
(() => {
  let loading;
  const players = new Set();
  function loadPlayer() {
    if (window.APlayer) return Promise.resolve();
    if (loading) return loading;
    loading = new Promise((resolve, reject) => {
      if (!document.getElementById('post-music-css')) {
        const css = document.createElement('link');
        css.id = 'post-music-css'; css.rel = 'stylesheet';
        css.href = 'https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.css';
        document.head.append(css);
      }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.js';
      script.async = true;
      const timer = setTimeout(() => reject(new Error('播放器加载超时')), 15000);
      script.onload = () => { clearTimeout(timer); window.APlayer ? resolve() : reject(new Error('播放器不可用')); };
      script.onerror = () => { clearTimeout(timer); reject(new Error('播放器加载失败')); };
      document.head.append(script);
    });
    return loading;
  }
  async function init() {
    const blocks = [...document.querySelectorAll('.post-body music-placeholder')].filter(el => !el.dataset.initialized);
    if (!blocks.length) return;
    blocks.forEach(el => { el.dataset.initialized = 'true'; });
    let enhanced = true;
    try { await loadPlayer(); } catch { enhanced = false; }
    for (const block of blocks) {
      if (!block.isConnected) continue;
      const d = block.dataset;
      const container = document.createElement('div');
      container.className = 'post-music';
      block.replaceWith(container);
      if (enhanced) {
        try {
          const player = new window.APlayer({container,
            audio: [{url: d.url, name: d.name || 'Audio', artist: d.artist || '', cover: d.cover || ''}],
            theme: d.theme || '#448aff', loop: 'none', volume: 0.7, mutex: true,
            autoplay: false, preload: 'none'
          });
          players.add(player);
          continue;
        } catch { container.replaceChildren(); }
      }
      // Keep the track usable if the CDN or player is unavailable.
      const title = document.createElement('p');
      title.textContent = [d.name, d.artist].filter(Boolean).join(' — ');
      if (d.cover) {
        const cover = document.createElement('img');
        cover.src = d.cover; cover.alt = d.name || '音乐封面'; cover.className = 'no-lightbox';
        cover.width = 72; cover.height = 72; container.append(cover);
      }
      const audio = document.createElement('audio');
      audio.controls = true; audio.preload = 'none'; audio.src = d.url;
      container.append(title, audio);
    }
  }
  function cleanup() {
    players.forEach(player => player.destroy()); players.clear();
    document.querySelectorAll('.post-music audio').forEach(audio => audio.pause());
  }
  document.addEventListener('pjax:send', cleanup);
  document.addEventListener('pjax:success', init);
  window.addEventListener('pagehide', () => {
    players.forEach(player => player.pause());
    document.querySelectorAll('.post-music audio').forEach(audio => audio.pause());
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
