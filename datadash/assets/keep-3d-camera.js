// Keep a 3D scene's camera where the user left it when an animation plays.
//
// Plotly saves the camera a drag leaves into the layout object the scene's GL
// plot was set up with (gl3d Scene.initializeGLPlot keeps the gd.layout of the
// time, in Plotly 4.1). Plotly.react replaces gd.layout but keeps the scene,
// and Dash's dcc.Graph calls it after the first camera change and whenever its
// figure changes, so from then on drags are saved into a layout nothing reads.
// An animation frame drawn with redraw, as a 3D animation's are, draws from
// gd.layout, and so jumps back to the camera it holds: the figure's own, or
// the first drag's. The scene still reports each camera in plotly_relayout, so
// copy it into gd.layout.

(function () {
  "use strict";

  var SCENE_CAMERA = /^(scene\d*)\.camera$/;

  function keepCamera(gd) {
    if (gd._keepsSceneCamera || typeof gd.on !== "function") {
      return;
    }
    gd._keepsSceneCamera = true;
    gd.on("plotly_relayout", function (update) {
      if (!update || !gd.layout) {
        return;
      }
      Object.keys(update).forEach(function (key) {
        var match = SCENE_CAMERA.exec(key);
        if (match) {
          var scene = gd.layout[match[1]] || (gd.layout[match[1]] = {});
          scene.camera = JSON.parse(JSON.stringify(update[key]));
        }
      });
    });
  }

  // Dash adds and replaces graphs as the page changes; Plotly marks each graph
  // div with js-plotly-plot once it has drawn it, and its events are set up by then
  function keepCameras(root) {
    if (root.classList && root.classList.contains("js-plotly-plot")) {
      keepCamera(root);
    }
    if (root.querySelectorAll) {
      root.querySelectorAll(".js-plotly-plot").forEach(keepCamera);
    }
  }

  function watch() {
    keepCameras(document.body);
    new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "attributes") {
          keepCameras(mutation.target);
        } else {
          mutation.addedNodes.forEach(keepCameras);
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
