// Fit a chart's title to the chart's width.
//
// Plotly draws a title on one line, however long, centred on the chart, and
// clips whatever runs past the chart's edges: on a phone, whose charts are
// about 300px wide, a title of more than about 23 characters at the theme's
// 20px lost both its ends. Once a chart has drawn, a title or subtitle wider
// than the chart, less a gap at each side, is drawn at the largest font size
// that fits (never smaller than MIN_SIZE). A narrower chart's title is
// smaller, never on more lines: Plotly lifts a title of several lines off the
// chart's top when the title reserves its band of the top margin, as the
// theme's do.
//
// The band keeps its depth: its pads take up the height the smaller text
// gives up, so the legend and the plot area stay where they were, in line with
// those of the charts beside it, and the title keeps its middle where it was.
// A title that reserves no band keeps its top where its figure put it. When
// the chart is wide enough again, its title gets its own size back.
//
// A chart's own sizes and pads are those it was drawn with, unless they are
// the ones this script gave it: Dash draws each new figure with its own, and
// redraws an unchanged one with a copy of its layout that keeps what it was
// given.

(function () {
  "use strict";

  // Room left between a title and each side of its chart, in px
  var GAP = 8;
  // The smallest font size a title is drawn at to fit, in px
  var MIN_SIZE = 10;
  // Font sizes are tried in steps of this, in px
  var STEP = 0.5;

  function titleOf(gd) {
    var title = gd._fullLayout && gd._fullLayout.title;
    return title && title.text ? title : null;
  }

  function subtitleOf(gd) {
    return gd.querySelector(".g-gtitle .gtitle-subtitle");
  }

  // The font sizes and pads the chart's title is drawn with
  function drawnWith(gd, title) {
    return {
      size: title.font.size,
      subtitle: subtitleOf(gd) ? title.subtitle.font.size : null,
      t: title.pad.t,
      b: title.pad.b,
    };
  }

  function same(a, b) {
    var close = function (x, y) {
      return x === y || (x !== null && y !== null && Math.abs(x - y) < 0.01);
    };
    return (
      Boolean(a && b) &&
      close(a.size, b.size) &&
      close(a.subtitle, b.subtitle) &&
      close(a.t, b.t) &&
      close(a.b, b.b)
    );
  }

  // The box of a title's text at a font size, measured on a copy of it next to
  // it: Plotly sets its lines' offsets in ems, so they scale with the copy
  function measure(node, size) {
    var copy = node.cloneNode(true);
    copy.style.fontSize = size + "px";
    copy.setAttribute("visibility", "hidden");
    node.parentNode.appendChild(copy);
    var box = copy.getBBox();
    node.parentNode.removeChild(copy);
    return { width: box.width, height: box.height, top: box.y };
  }

  // The largest font size, up to own, at which the text fits in width, with
  // the text's box at own size and at that size
  function fit(node, own, width) {
    var full = measure(node, own);
    var size = own;
    var box = full;
    if (box.width > width) {
      // Text widens about in proportion to its font size; step down from
      // there to the first size that fits
      size = Math.max(MIN_SIZE, Math.floor((own * width) / box.width / STEP) * STEP);
      box = measure(node, size);
      while (box.width > width && size > MIN_SIZE) {
        size = Math.max(MIN_SIZE, size - STEP);
        box = measure(node, size);
      }
    }
    return { size: size, own: full, now: box };
  }

  // The font sizes and pads the title fits the chart with, from its own
  function fitted(gd, title, own) {
    var width = gd._fullLayout.width - 2 * GAP;
    var main = fit(gd.querySelector(".g-gtitle .gtitle"), own.size, width);
    var sub = null;
    var subtitle = subtitleOf(gd);
    if (own.subtitle !== null && subtitle) {
      sub = fit(subtitle, own.subtitle, width);
    }
    // How far the smaller title's text moves down from where its own was, at
    // its top and at its middle
    var top = main.now.top - main.own.top;
    var middle = top + (main.now.height - main.own.height) / 2;
    if (!title.automargin) {
      // Its top where its figure put it
      return {
        size: main.size,
        subtitle: sub && sub.size,
        t: Math.max(0, own.t - top),
        b: own.b,
      };
    }
    // Its middle where it was, in a band as deep as it was: the pads take up
    // what the smaller text gives up
    var lost = main.own.height - main.now.height;
    if (sub) {
      lost += sub.own.height - sub.now.height;
    }
    var t = Math.max(0, own.t - middle);
    return {
      size: main.size,
      subtitle: sub && sub.size,
      t: t,
      b: Math.max(0, own.t + own.b + lost - t),
    };
  }

  function afterPlot(gd) {
    var title = titleOf(gd);
    if (gd._fittingTitle || !title || !gd.querySelector(".g-gtitle .gtitle")) {
      return;
    }
    if (gd.getBoundingClientRect().width === 0) {
      return; // on a hidden tab, drawn at no width of its own
    }
    var now = drawnWith(gd, title);
    if (!same(now, gd._titleGiven)) {
      // Drawn with its figure's own
      gd._titleOwn = now;
    }
    var own = gd._titleOwn;
    var want = fitted(gd, title, own);
    if (same(want, now)) {
      return;
    }
    gd._titleGiven = same(want, own) ? null : want;
    var update = {
      "title.font.size": want.size,
      // Set whole: Plotly 4.1 leaves a title's band as it was when one of its
      // pads is set alone
      "title.pad": { t: want.t, b: want.b, l: title.pad.l, r: title.pad.r },
    };
    if (want.subtitle !== null) {
      update["title.subtitle.font.size"] = want.subtitle;
    }
    gd._fittingTitle = true;
    var done = function () {
      gd._fittingTitle = false;
    };
    window.Plotly.relayout(gd, update).then(done, done);
  }

  // Dash adds and replaces graphs as the page changes; Plotly marks each graph
  // div with js-plotly-plot when it starts drawing it
  function watchChart(gd) {
    if (gd._fitsTitle || typeof gd.on !== "function") {
      return;
    }
    gd._fitsTitle = true;
    gd.on("plotly_afterplot", function () {
      afterPlot(gd);
    });
    if (gd._fullLayout && gd._fullLayout._size) {
      afterPlot(gd);
    }
  }

  function watchCharts(root) {
    if (root.classList && root.classList.contains("js-plotly-plot")) {
      watchChart(root);
    }
    if (root.querySelectorAll) {
      root.querySelectorAll(".js-plotly-plot").forEach(watchChart);
    }
  }

  function watch() {
    watchCharts(document.body);
    new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "attributes") {
          watchCharts(mutation.target);
        } else {
          mutation.addedNodes.forEach(watchCharts);
        }
      });
    }).observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["class"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watch);
  } else {
    watch();
  }
})();
