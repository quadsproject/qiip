// Shared USD formatting for the premium-equivalent cost estimates. Pinned to
// en-US so every browser shows "$9.00" regardless of locale, and a missing
// amount renders as $0.00 rather than $NaN.

(function () {
  "use strict";

  window.qiipFormatUsd = function (amount) {
    return "$" + (Number(amount) || 0).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  };
})();
