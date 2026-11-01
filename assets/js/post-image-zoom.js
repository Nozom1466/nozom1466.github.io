// Replace the theme's gallery viewer before NexT boots. No carousel or navigation.
(() => {
  let dialog, largeImage, opener, previousOverflow;
  function close() { if (dialog?.open) dialog.close(); }
  function open(image) {
    if (!dialog) {
      dialog = document.createElement('dialog');
      dialog.id = 'post-image-zoom';
      dialog.setAttribute('aria-label', '图片大图，点击空白区域或按 Esc 关闭');
      largeImage = document.createElement('img');
      dialog.append(largeImage);
      document.body.append(dialog);
      dialog.addEventListener('click', event => { if (event.target === dialog) close(); });
      dialog.addEventListener('close', () => {
        document.body.style.overflow = previousOverflow;
        largeImage.removeAttribute('src');
        if (opener?.isConnected) opener.focus({preventScroll:true});
      });
      document.addEventListener('pjax:send', close);
    }
    opener = image.closest('a') || image;
    previousOverflow = document.body.style.overflow;
    largeImage.alt = image.alt || '文章配图';
    largeImage.src = image.dataset.src || image.currentSrc || image.src;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  }
  NexT.utils.registerImageViewer = function () {
    document.querySelectorAll('.post-body').forEach(body => {
      if (body.dataset.simpleZoom) return;
      body.dataset.simpleZoom = 'true';
      body.querySelectorAll('img:not(.no-lightbox)').forEach(image => {
        if (!image.closest('a')) { image.tabIndex = 0; image.setAttribute('role', 'button'); }
      });
      body.addEventListener('click', event => {
        const image = event.target.closest('img:not(.no-lightbox)');
        if (!image || !body.contains(image)) return;
        // Prevent the original-image link from opening a download/new tab.
        event.preventDefault();
        event.stopImmediatePropagation();
        open(image);
      }, true);
      body.addEventListener('keydown', event => {
        const target = event.target;
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const image = target.matches('img:not(.no-lightbox)') ? target : target.matches('a') ? target.querySelector('img:not(.no-lightbox)') : null;
        if (!image) return;
        event.preventDefault();
        open(image);
      });
    });
  };
})();
