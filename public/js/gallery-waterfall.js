// Gallery Waterfall Layout
// Implements a justified/waterfall layout for gallery images

(function() {
    'use strict';

    const gallery = document.getElementById('gallery-container');
    if (!gallery) return;

    const items = Array.from(gallery.querySelectorAll('.gallery-item'));
    if (items.length === 0) return;

    // Configuration
    const config = {
        spacing: 8,          // Gap between images
        targetRowHeight: 300, // Target height for rows
        tolerance: 0.25       // Height tolerance (0-1)
    };

    let containerWidth = 0;

    // Calculate layout
    function calculateLayout() {
        const newWidth = gallery.offsetWidth;
        if (newWidth === containerWidth) return;
        containerWidth = newWidth;

        // Get aspect ratios
        const aspectRatios = items.map(item => {
            const width = parseFloat(item.dataset.width);
            const height = parseFloat(item.dataset.height);
            return width / height;
        });

        // Simple waterfall algorithm
        const rows = [];
        let currentRow = [];
        let currentRowAspectRatio = 0;

        aspectRatios.forEach((ratio, index) => {
            currentRow.push({ index, ratio });
            currentRowAspectRatio += ratio;

            // Calculate if row should be completed
            const rowWidth = containerWidth - (currentRow.length - 1) * config.spacing;
            const calculatedHeight = rowWidth / currentRowAspectRatio;
            
            const minHeight = config.targetRowHeight * (1 - config.tolerance);
            const maxHeight = config.targetRowHeight * (1 + config.tolerance);

            // Complete row if height is in acceptable range or it's the last item
            if ((calculatedHeight >= minHeight && calculatedHeight <= maxHeight) || 
                index === aspectRatios.length - 1) {
                rows.push({
                    items: currentRow,
                    aspectRatio: currentRowAspectRatio,
                    height: calculatedHeight
                });
                currentRow = [];
                currentRowAspectRatio = 0;
            }
        });

        // Position items
        let currentTop = 0;
        
        rows.forEach(row => {
            const rowWidth = containerWidth - (row.items.length - 1) * config.spacing;
            const rowHeight = rowWidth / row.aspectRatio;
            let currentLeft = 0;

            row.items.forEach(({ index, ratio }) => {
                const itemWidth = ratio * rowHeight;
                const item = items[index];

                item.style.position = 'absolute';
                item.style.top = currentTop + 'px';
                item.style.left = currentLeft + 'px';
                item.style.width = itemWidth + 'px';
                item.style.height = rowHeight + 'px';

                currentLeft += itemWidth + config.spacing;
            });

            currentTop += rowHeight + config.spacing;
        });

        // Set container height
        gallery.style.height = currentTop + 'px';
        gallery.style.position = 'relative';
    }

    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Initialize
    const debouncedLayout = debounce(calculateLayout, 150);
    
    // Calculate on load and resize
    window.addEventListener('resize', debouncedLayout);
    window.addEventListener('orientationchange', debouncedLayout);

    // Initial calculation after images load
    let loadedImages = 0;
    const totalImages = items.length;

    items.forEach(item => {
        const img = item.querySelector('img');
        if (img) {
            if (img.complete) {
                loadedImages++;
                if (loadedImages === totalImages) {
                    calculateLayout();
                }
            } else {
                img.addEventListener('load', () => {
                    loadedImages++;
                    if (loadedImages === totalImages) {
                        calculateLayout();
                    }
                });
            }
        }
    });

    // Fallback if images don't trigger load
    setTimeout(() => {
        calculateLayout();
    }, 500);
})();
