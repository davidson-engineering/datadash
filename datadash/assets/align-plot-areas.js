// Line up the plot areas of the charts stacked in one column of the page, and
// give a narrow chart, a phone's, as wide a plot area as its labels allow.
//
// Plotly sizes each chart's margins from its own tick labels and axis titles
// (automargin), so charts of the same width stacked in a grid column have plot
// areas whose edges are as far apart as their labels' widths differ. Plotly
// lays out each chart on its own, so once a chart has drawn, give every chart
// in its column the widest left and right margins any of them needs. Those are
// only the charts' base margins: automargin still widens a margin a chart's
// labels need, so a label is never clipped.
//
// The theme's labels and margins are sized for a desktop's charts: a phone's,
// about 300px wide, kept a plot area of 123px between margins of 90px, the
// theme's 70px widened by its 14px tick labels and axis titles. A chart
// narrower than COMPACT_WIDTH is drawn compact: its axes' labels and titles
// smaller, each title closer to its labels, and base margins of COMPACT_MARGIN,
// so automargin makes its margins what its labels need and no more. A wider
// chart is drawn with its figure's own style.
//
// What a chart needs is what Plotly made of that base style (_fullLayout._size).
// Once it has been given wider margins, its width changing can change its tick
// labels, so it is drawn with its base style again to measure them.
//
// The script relayouts charts without Dash's graph hearing of each step, and
// tells it of the style each settles on: the graph keeps that in the figure it
// draws the chart from again. A chart drawn from a figure without the mark the
// style carries is a new figure, drawn with its own style.

(function () {
  "use strict";

  // The layout key that marks a figure Dash's graph made from a layout styled
  // here, as against a new one
  var MARK = "datadashPlotArea";
  // Charts narrower than this, in px, are drawn compact: a phone's (up to
  // about 360px on a 430px screen), not those two to a row on a 1024px screen
  // (464px)
  var COMPACT_WIDTH = 420;
  // A compact chart's base left and right margins, in px
  var COMPACT_MARGIN = 10;
  // A compact chart's axis label and title font sizes, as a share of its own,
  // and the smallest it shrinks them to, in px
  var COMPACT_FONT = 0.8;
  var MIN_FONT = 10;
  // The most room a compact chart leaves between an axis title and its tick
  // labels, in px (Plotly's default is 15)
  var COMPACT_STANDOFF = 5;

  // Pixel margins of the plot area Plotly drew
  function drawn(gd) {
    var size = gd._fullLayout._size;
    return { l: size.l, r: size.r };
  }

  function same(a, b) {
    return Boolean(a && b) && a.l === b.l && a.r === b.r;
  }

  function isCartesian(gd) {
    var full = gd._fullLayout;
    return Boolean(full && full._size && full._has && full._has("cartesian"));
  }

  function valueAt(layout, key) {
    return key.split(".").reduce(function (value, name) {
      return value === null || value === undefined ? undefined : value[name];
    }, layout);
  }

  // The style the chart is drawn with, by layout key: its base left and right
  // margins, which automargin widens, and its axes' tick label and title font
  // sizes and title standoffs
  function styleOf(gd) {
    var full = gd._fullLayout;
    var keys = ["margin.l", "margin.r"];
    full._subplots.xaxis.concat(full._subplots.yaxis).forEach(function (id) {
      var axis = id.charAt(0) + "axis" + id.slice(1);
      keys.push(axis + ".tickfont.size", axis + ".title.font.size", axis + ".title.standoff");
    });
    var style = {};
    keys.forEach(function (key) {
      var value = valueAt(full, key);
      if (typeof value === "number") {
        style[key] = value;
      }
    });
    return style;
  }

  // Whether the chart is drawn with the style (none: never)
  function drawnWith(gd, style) {
    return (
      Boolean(style) &&
      Object.keys(style).every(function (key) {
        var value = valueAt(gd._fullLayout, key);
        return typeof value === "number" && Math.abs(value - style[key]) < 0.01;
      })
    );
  }

  function compact(own) {
    var style = {};
    Object.keys(own).forEach(function (key) {
      var value = own[key];
      if (key.indexOf("margin.") === 0) {
        style[key] = Math.min(value, COMPACT_MARGIN);
      } else if (/\.standoff$/.test(key)) {
        style[key] = Math.min(value, COMPACT_STANDOFF);
      } else {
        style[key] = Math.max(Math.min(value, MIN_FONT), Math.round(value * COMPACT_FONT));
      }
    });
    return style;
  }

  // The style the chart is measured with at its width: compact or its own
  function baseStyle(gd) {
    var own = gd._alignOwn;
    return gd._fullLayout.width < COMPACT_WIDTH ? compact(own) : own;
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

  // Relayout the chart without Dash's graph hearing of it. After each
  // relayout it hears of, the graph draws the chart again from the figure it
  // holds, updated with the chart's layout under the keys the relayout named
  // as they were then: it would draw a column again with each step it takes
  // to settle, after it had settled. It hears of the style a chart settles on
  // once (tell)
  function relayoutQuietly(gd, update) {
    var emit = gd.emit;
    gd.emit = function (name, data) {
      var ours =
        name === "plotly_relayout" &&
        Boolean(data) &&
        Object.keys(data).every(function (key) {
          return key in update;
        });
      if (!ours) {
        emit.apply(gd, arguments);
      } else if (gd._internalEv) {
        gd._internalEv.emit(name, data);
      }
    };
    var restore = function () {
      gd.emit = emit;
    };
    var relayout = window.Plotly.relayout(gd, update);
    relayout.then(restore, restore);
    return relayout;
  }

  // Tell Dash's graph of the style the chart has settled on, as if the chart
  // had been relayouted to it, so the figure it holds has it: it would draw
  // the chart with the figure's own otherwise, or, where another relayout
  // named an axis (a zoom), with that axis as styled here and the rest of the
  // chart not
  function tell(gd) {
    gd._alignUntold = false;
    var data = Object.assign({}, gd._alignGiven);
    data[MARK] = true;
    gd.emit("plotly_relayout", data);
  }

  // Draw the chart with the style, then call done. Its layout is marked as
  // styled here, so a figure Dash's graph makes from it is told apart from a
  // new one
  function setStyle(gd, style, done) {
    var given = function () {
      gd._alignBusy = false;
      done();
    };
    gd._alignGiven = style;
    if (drawnWith(gd, style)) {
      given();
      return;
    }
    var update = Object.assign({}, style);
    update[MARK] = true;
    gd._alignBusy = true;
    gd._alignUntold = true;
    relayoutQuietly(gd, update).then(given, given);
  }

  // Automargin widens a margin past the one given when the chart's labels have
  // changed and need more (a trace shown, say, or its plot area narrowed by
  // the margins it was given): then that is what it needs
  function outgrown(gd, given) {
    var now = drawn(gd);
    if (now.l > given["margin.l"]) {
      gd._alignNeed.l = Math.max(gd._alignNeed.l, now.l);
    }
    if (now.r > given["margin.r"]) {
      gd._alignNeed.r = Math.max(gd._alignNeed.r, now.r);
    }
  }

  // Charts being measured, and the frame's pass over them: columns are aligned
  // once neither is under way
  var measuring = 0;

  function measured() {
    measuring -= 1;
    if (measuring === 0) {
      align();
    }
  }

  // Draw the chart with its base style, if it is not, and measure what it needs
  function measure(gd) {
    measuring += 1;
    setStyle(gd, baseStyle(gd), function () {
      gd._alignNeed = Object.assign(drawn(gd), { width: gd._fullLayout.width });
      measured();
    });
  }

  // Give every chart in each column the widest margins any of them needs. Once
  // a chart has drawn with them, its column is aligned again at once: it may
  // have outgrown them, and a chart still drawing was passed over. So a column
  // given margins in a frame is drawn with those it settles on before the
  // browser paints the frame
  function align() {
    if (measuring > 0) {
      return;
    }
    var all = document.querySelectorAll(".js-plotly-plot");
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
        var style = Object.assign({}, baseStyle(gd), {
          "margin.l": widest.l,
          "margin.r": widest.r,
        });
        setStyle(gd, style, function () {
          outgrown(gd, style);
          align();
        });
      });
    });
    // Settled: tell Dash's graph of the charts' styles
    var settled = Array.prototype.every.call(all, function (gd) {
      return !gd._alignBusy;
    });
    if (settled) {
      Array.prototype.forEach.call(all, function (gd) {
        if (gd._alignUntold) {
          tell(gd);
        }
      });
    }
  }

  // Measure the charts drawn since the last frame and align the columns,
  // before the browser paints the frame they appear in, so no chart is ever
  // seen with its own style and then changes. Dash draws a page's charts one
  // after another in one task, and its graph hears of a chart's relayouts once
  // it has drawn the chart: by the next frame, not before
  function frame() {
    measuring += 1;
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      if (!gd._alignOwn || gd._alignBusy || !isCartesian(gd)) {
        return;
      }
      var given = gd._alignGiven;
      if (!gd._alignNeed) {
        measure(gd);
      } else if (!drawnWith(gd, given)) {
        // Redrawn by Dash from a figure without the style last given here
        setStyle(gd, given, function () {
          outgrown(gd, given);
          align();
        });
      } else {
        outgrown(gd, given);
      }
    });
    measured();
  }

  var pending = null;
  function schedule() {
    if (pending === null) {
      pending = window.requestAnimationFrame(function () {
        pending = null;
        frame();
      });
    }
  }

  function afterPlot(gd) {
    if (gd._alignBusy || !isCartesian(gd)) {
      return;
    }
    if (!gd.layout[MARK] || !gd._alignGiven) {
      // Drawn with its figure's own style
      gd._alignOwn = styleOf(gd);
      gd._alignGiven = null;
      gd._alignNeed = null;
    } else if (gd._alignNeed && gd._alignNeed.width !== gd._fullLayout.width) {
      // Resized, to be measured again
      gd._alignNeed = null;
    }
    schedule();
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
