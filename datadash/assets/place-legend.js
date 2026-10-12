// Place a chart's legend again where Plotly left it out of place.
//
// The theme's title reserves its band of the top margin (title.automargin),
// and a legend over the plot area is pushed out below that band. On a chart's
// first draw, Plotly 4.1 places the legend before the title has reserved its
// band, then moves the plot area down for the band without placing the legend
// again: subroutines.layoutStyles applies the band while the chart is still
// being drawn, so no redraw follows. The legend is left over the title. Plotly
// keeps the band from one draw to the next, so placing the legend again, once
// the chart has drawn, puts it under the title.

(function () {
  "use strict";

  // How far a legend's top is above its y, as a fraction of its height
  var ABOVE_Y = { top: 0, middle: 0.5, bottom: 1 };

  // The anchor Plotly takes a legend's yanchor "auto" to mean (legend/helpers.js)
  function yanchor(legend) {
    if (legend.yanchor !== "auto") {
      return legend.yanchor;
    }
    return legend.y <= 1 / 3 ? "bottom" : legend.y >= 2 / 3 ? "top" : "middle";
  }

  // Whether the legend is where the chart's margins put it, as Plotly places
  // a legend on the paper (legend/draw.js)
  function placed(gd) {
    var full = gd._fullLayout;
    var legend = full && full.legend;
    var node = gd.querySelector(".legend");
    if (!full || !full.showlegend || !legend || legend.yref !== "paper" || !node) {
      return true;
    }
    var size = full._size;
    var top = size.t + size.h * (1 - legend.y) - ABOVE_Y[yanchor(legend)] * legend._effHeight;
    if (full.margin.autoexpand) {
      top = Math.min(Math.max(top, 0), full.height - legend._effHeight);
    }
    var at = /translate\(\s*([-\d.e]+)[\s,]+([-\d.e]+)/.exec(node.getAttribute("transform") || "");
    return !at || Math.abs(parseFloat(at[2]) - top) < 0.5;
  }

  function afterPlot(gd) {
    if (gd._placingLegend || placed(gd)) {
      return;
    }
    // Relaying out the legend as it is places it again, and only it
    gd._placingLegend = true;
    var done = function () {
      gd._placingLegend = false;
    };
    window.Plotly.relayout(gd, { "legend.y": gd._fullLayout.legend.y }).then(done, done);
  }

  // Dash adds and replaces graphs as the page changes; Plotly marks each graph
  // div with js-plotly-plot when it starts drawing it
  function watchChart(gd) {
    if (gd._placesLegend || typeof gd.on !== "function") {
      return;
    }
    gd._placesLegend = true;
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
