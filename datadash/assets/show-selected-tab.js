// Keep the selected tab in view in a tab bar that scrolls.
//
// Where fewer than three of create_tabs's tabs fit side by side, they keep to
// one row that scrolls sideways (dash-components.css), so the selected tab can
// start out of view, as when a dashboard opens on a tab other than its first.
// Whenever a bar's selected tab changes, or the bar is resized, scroll the bar
// to centre that tab if it is not wholly in view.

(function () {
  "use strict";

  var BAR = "datadash-tabs";

  function showSelectedTab(bar, behavior) {
    var tab = bar.querySelector(":scope > .tab--selected");
    if (!tab || bar.scrollWidth <= bar.clientWidth) {
      return;
    }
    var barBox = bar.getBoundingClientRect();
    var tabBox = tab.getBoundingClientRect();
    var left = tabBox.left - barBox.left - bar.clientLeft;
    if (left >= 0 && left + tabBox.width <= bar.clientWidth) {
      return;
    }
    bar.scrollTo({
      left: bar.scrollLeft + left - (bar.clientWidth - tabBox.width) / 2,
      behavior: behavior,
    });
  }

  function watchBar(bar) {
    if (bar._showsSelectedTab) {
      return;
    }
    bar._showsSelectedTab = true;
    new ResizeObserver(function () {
      showSelectedTab(bar, "instant");
    }).observe(bar);
  }

  // Dash adds the tab bars as it renders the page, and marks the selected tab
  // with tab--selected
  function watchBars(root) {
    if (!root.classList) {
      return;
    }
    if (root.classList.contains(BAR)) {
      watchBar(root);
    }
    root.querySelectorAll("." + BAR).forEach(watchBar);
  }

  function watch() {
    watchBars(document.body);
    new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "attributes") {
          var bar = mutation.target.parentElement;
          if (
            mutation.target.classList.contains("tab--selected") &&
            bar &&
            bar.classList.contains(BAR)
          ) {
            showSelectedTab(bar, "smooth");
          }
        } else {
          mutation.addedNodes.forEach(watchBars);
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
