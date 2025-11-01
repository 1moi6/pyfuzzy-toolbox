// Open external links in new tab
document.addEventListener('DOMContentLoaded', function() {
  var links = document.querySelectorAll('a');

  links.forEach(function(link) {
    var href = link.getAttribute('href');

    // Check if link is external (starts with http/https and not current domain)
    if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
      // Don't modify if it's a link to the same domain
      if (!href.includes(window.location.hostname)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    }
  });
});
