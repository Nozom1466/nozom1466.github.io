import justifiedLayout from "./justified-layout.js";
import * as params from "@params";

const gallery = document.getElementById("gallery");

if (gallery) {
  let containerWidth = 0;
  const allItems = gallery.querySelectorAll(".gallery-item");

  // Apply tag filtering BEFORE first layout
  const urlParams = new URLSearchParams(window.location.search);
  const currentTag = urlParams.get('tag');
  if (currentTag) {
    allItems.forEach(item => {
      const itemTags = (item.getAttribute('data-tags') || '').split(',');
      if (!itemTags.includes(currentTag)) {
        item.classList.add('tag-hidden');
        item.style.display = 'none';
      }
    });
  }

  function getVisibleItems() {
    return Array.from(allItems).filter(item => !item.classList.contains('tag-hidden'));
  }

  // Load image and get natural dimensions
  function loadImageDimensions(item) {
    return new Promise((resolve) => {
      // Get original image URL from parent <a> element's data-pswp-src
      const originalSrc = item.getAttribute('data-pswp-src') || item.href;
      
      if (!originalSrc) {
        // Fallback to default dimensions
        resolve({ width: 1200, height: 800 });
        return;
      }

      const tempImg = new Image();
      tempImg.onload = function() {
        resolve({
          width: this.naturalWidth,
          height: this.naturalHeight
        });
      };
      tempImg.onerror = function() {
        // Fallback on error
        resolve({ width: 1200, height: 800 });
      };
      tempImg.src = originalSrc;
    });
  }

  async function updateGallery(force) {
    if (!force && containerWidth === gallery.getBoundingClientRect().width) return;
    containerWidth = gallery.getBoundingClientRect().width;

    const visibleItems = getVisibleItems();
    
    // Hide filtered items completely
    allItems.forEach(item => {
      if (item.classList.contains('tag-hidden')) {
        item.style.display = 'none';
      } else {
        item.style.display = '';
      }
    });

    if (visibleItems.length === 0) {
      gallery.style.height = "100px";
      return;
    }

    // Load all images and get their dimensions
    const dimensionsPromises = visibleItems.map(async (item) => {
      const dims = await loadImageDimensions(item);
      const img = item.querySelector("img");
      const figure = item.querySelector("figure");
      const aspectRatio = dims.width / dims.height;
      
      // Update img attributes with actual dimensions
      img.setAttribute('width', dims.width);
      img.setAttribute('height', dims.height);
      
      // Update PhotoSwipe data attributes on the parent <a> element
      item.setAttribute('data-pswp-width', dims.width);
      item.setAttribute('data-pswp-height', dims.height);
      
      // Update aspect-ratio styles
      item.style.aspectRatio = aspectRatio;
      if (figure) {
        figure.style.aspectRatio = aspectRatio;
      }
      
      return dims;
    });

    const dimensions = await Promise.all(dimensionsPromises);
    const aspectRatios = dimensions.map(d => d.width / d.height);

    visibleItems.forEach((item) => {
      const img = item.querySelector("img");
      img.style.width = "100%";
      img.style.height = "auto";
    });

    const layout = justifiedLayout(aspectRatios, {
      rowWidth: containerWidth,
      spacing: Number.isInteger(params.boxSpacing) ? params.boxSpacing : 8,
      rowHeight: params.targetRowHeight || 288,
      heightTolerance: Number.isInteger(params.targetRowHeightTolerance) ? params.targetRowHeightTolerance : 0.25,
    });

    visibleItems.forEach((item, i) => {
      const { width, height, top, left } = layout.boxes[i];
      item.style.position = "absolute";
      item.style.width = width + "px";
      item.style.height = height + "px";
      item.style.top = top + "px";
      item.style.left = left + "px";
      item.style.overflow = "hidden";
    });

    gallery.style.position = "relative";
    gallery.style.height = layout.containerHeight + "px";
    gallery.style.visibility = "";
  }

  // Expose updateGallery globally for tag filtering
  window.updateGalleryLayout = function() {
    containerWidth = 0; // Force recalculation
    updateGallery(true);
  };

  window.addEventListener("resize", () => updateGallery(false));
  window.addEventListener("orientationchange", () => updateGallery(false));

  // Call twice to adjust for scrollbars appearing after first call
  updateGallery(true).then(() => {
    updateGallery(true).then(() => {
      // Hide loading indicator
      const loadingEl = document.getElementById('gallery-loading');
      if (loadingEl) {
        loadingEl.classList.add('hidden');
        setTimeout(() => {
          loadingEl.style.display = 'none';
        }, 300);
      }
      // Dispatch event after images are loaded and dimensions updated
      window.dispatchEvent(new CustomEvent('galleryReady'));
    });
  });
}
