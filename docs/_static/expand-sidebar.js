// Furo's sidebar uses CSS-only collapse via hidden <input type="checkbox"
// class="toctree-checkbox"> elements paired with adjacent <label>s and a
// `:checked ~ ul` rule. To force every branch open, just check every
// toctree checkbox on page load.
//
// Exception: branches under the captions in SKIP_CAPTIONS stay collapsed
// (because they're enormous — e.g. the autodoc'd API tree spans hundreds
// of pages and would dwarf the hand-written sections).

const SKIP_CAPTIONS = new Set(["API reference"]);

function _expandFuroSidebar() {
  // Identify the <ul> subtrees whose captions should remain collapsed.
  const skipUls = new Set();
  document
    .querySelectorAll(".sidebar-tree p.caption")
    .forEach((p) => {
      const text = (
        p.querySelector(".caption-text")?.textContent ?? ""
      ).trim();
      if (!SKIP_CAPTIONS.has(text)) return;
      // The caption's toctree is the next sibling <ul>.
      const ul = p.nextElementSibling;
      if (ul && ul.tagName === "UL") skipUls.add(ul);
    });

  document
    .querySelectorAll(".sidebar-tree .toctree-checkbox")
    .forEach((cb) => {
      // Skip checkboxes that live anywhere inside a skipped subtree.
      for (let p = cb.parentElement; p; p = p.parentElement) {
        if (skipUls.has(p)) return;
      }
      cb.checked = true;
    });
}

document.addEventListener("DOMContentLoaded", () => {
  _expandFuroSidebar();
  // Safety: re-run after the theme's own scripts settle.
  setTimeout(_expandFuroSidebar, 0);
});
