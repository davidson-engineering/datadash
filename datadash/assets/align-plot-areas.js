// Line up the plot areas of the charts stacked in one column of the page.
//
// Plotly sizes each chart's margins from its own tick labels and axis titles
// (automargin), so charts of the same width stacked in a grid column have plot
// areas whose edges are as far apart as their labels' widths differ. Plotly
// lays out each chart on its own, so once a chart has drawn, give every chart
// in its column the widest left and right margins any of them needs. Those are
// only the charts' base margins: automargin still widens a margin a chart's
// labels need, so a label is never clipped.
//
// What a chart needs is what Plotly made of its figure's own margins
// (_fullLayout._size). Once it has been given wider ones, its width changing
// can change its tick labels, so it is drawn with its own margins again to
// measure them. A chart has been given margins while it is drawn with the ones
// it was given: Dash draws each new figure with that figure's own, and redraws
// an unchanged one with a copy of its layout that keeps the margins given to it.

(function () {
  "use strict";

  // Pixel margins of the plot area Plotly drew
  function drawn(gd) {
    var size = gd._fullLayout._size;
    return { l: size.l, r: size.r };
  }

  // Base margins the chart was drawn with, which automargin widens
  function base(gd) {
    var margin = gd._fullLayout.margin;
    return { l: margin.l, r: margin.r };
  }

  function same(a, b) {
    return Boolean(a && b) && a.l === b.l && a.r === b.r;
  }

  function isCartesian(gd) {
    var full = gd._fullLayout;
    return Boolean(full && full._size && full._has && full._has("cartesian"));
  }

  // The charts with a measured need at their current width, by column: charts
  // in one column have the same left edge and width
  function columns() {
    var byColumn = {};
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      var box = gd.getBoundingClientRect();
      if (!gd._alignNeed || box.width === 0 || !isCartesian(gd)) {
        return;
      }
      if (gd._alignNeed.width !== gd._fullLayout.width) {
        return; // resized, and being measured again
      }
      var key = Math.round(box.left) + ":" + Math.round(box.width);
      (byColumn[key] = byColumn[key] || []).push(gd);
    });
    return Object.keys(byColumn).map(function (key) {
      return byColumn[key];
    });
  }

  // Draw the chart with base margins {l, r}, then call done
  function setMargins(gd, margins, done) {
    gd._alignBusy = true;
    window.Plotly.relayout(gd, { "margin.l": margins.l, "margin.r": margins.r }).then(
      function () {
        gd._alignBusy = false;
        gd._alignGiven = same(margins, gd._alignOwn) ? null : margins;
        done();
      },
      function () {
        gd._alignBusy = false;
      }
    );
  }

  // Automargin widens a margin past the one given when the chart's labels have
  // changed and need more (a trace shown, say): then that is what it needs
  function outgrown(gd, given) {
    var now = drawn(gd);
    if (now.l > given.l || now.r > given.r) {
      gd._alignNeed.l = Math.max(gd._alignNeed.l, now.l);
      gd._alignNeed.r = Math.max(gd._alignNeed.r, now.r);
      schedule();
    }
  }

  function align() {
    columns().forEach(function (charts) {
      var widest = { l: 0, r: 0 };
      charts.forEach(function (gd) {
        widest.l = Math.max(widest.l, gd._alignNeed.l);
        widest.r = Math.max(widest.r, gd._alignNeed.r);
      });
      charts.forEach(function (gd) {
        if (gd._alignBusy || same(drawn(gd), widest)) {
          return;
        }
        setMargins(gd, widest, function () {
          outgrown(gd, widest);
        });
      });
    });
  }

  // Dash draws a page's charts one after another in one task; align them all
  // before the browser paints the frame they appear in, so no chart is ever
  // seen with its own margins and then moves
  var pending = null;
  function schedule() {
    if (pending === null) {
      pending = window.requestAnimationFrame(function () {
        pending = null;
        align();
      });
    }
  }

  function measure(gd) {
    gd._alignNeed = Object.assign(drawn(gd), { width: gd._fullLayout.width });
    schedule();
  }

  function afterPlot(gd) {
    if (gd._alignBusy || !isCartesian(gd)) {
      return;
    }
    if (!same(base(gd), gd._alignGiven)) {
      // Drawn with its figure's own margins, so with what it needs
      gd._alignOwn = base(gd);
      gd._alignGiven = null;
      measure(gd);
    } else if (gd._fullLayout.width !== gd._alignNeed.width) {
      setMargins(gd, gd._alignOwn, function () {
        measure(gd);
      });
    } else {
      outgrown(gd, gd._alignGiven);
    }
  }

  // Dash adds and replaces graphs as the page changes; Plotly marks each graph
  // div with js-plotly-plot when it starts drawing it
  function watchChart(gd) {
    if (gd._alignsPlotArea || typeof gd.on !== "function") {
      return;
    }
    gd._alignsPlotArea = true;
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
