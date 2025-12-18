/* custom.js */

// Show/hide PhotoSwipe arrows based on mouse position (edge detection)
document.addEventListener('DOMContentLoaded', function() {
  // Arrow button width threshold (pixels from edge)
  const edgeThreshold = 100;
  
  document.addEventListener('mousemove', function(e) {
    const pswp = document.querySelector('.pswp--open');
    if (!pswp) return;
    
    const prevArrow = pswp.querySelector('.pswp__button--arrow--prev');
    const nextArrow = pswp.querySelector('.pswp__button--arrow--next');
    const screenWidth = window.innerWidth;
    const mouseX = e.clientX;
    
    // Show left arrow when mouse is near left edge
    if (prevArrow) {
      if (mouseX <= edgeThreshold) {
        prevArrow.style.opacity = '1';
      } else {
        prevArrow.style.opacity = '0';
      }
    }
    
    // Show right arrow when mouse is near right edge
    if (nextArrow) {
      if (mouseX >= screenWidth - edgeThreshold) {
        nextArrow.style.opacity = '1';
      } else {
        nextArrow.style.opacity = '0';
      }
    }
  });
});
