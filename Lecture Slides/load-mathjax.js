window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    svg: {
        fontCache: 'global'
    }
};

(function () {
    var script = document.createElement('script');
    script.src = '../MathJax/es5/tex-svg.js';
    script.async = true;
    document.head.appendChild(script);
})();