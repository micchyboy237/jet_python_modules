(() => {
  // Avoid overwriting if already loaded
  if (window.__utilsInjected__) return;
  window.__utilsInjected__ = true;

  /**
   * Simple utility namespace.
   * All helper functions live under `window.Utils`.
   */
  window.Utils = {
    /**
     * Get bounding box info for an element matching a selector.
     * @param {string} selector - The CSS selector for the element.
     * @returns {Object|null} bounding box {x, y, width, height, top, left, bottom, right}
     */
    getBoundingBox(selector) {
      const el = document.querySelector(selector);
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        top: rect.top,
        left: rect.left,
        bottom: rect.bottom,
        right: rect.right,
      };
    },

    /**
     * Scroll element into view smoothly.
     * @param {string} selector - CSS selector for the target element.
     * @returns {boolean} true if scrolled successfully, false otherwise.
     */
    scrollIntoView(selector) {
      const el = document.querySelector(selector);
      if (!el) return false;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      return true;
    },

    /**
     * Get all text content of leaf nodes inside a given element.
     * @param {string} selector - CSS selector for the parent element.
     * @returns {string[]} Array of leaf text contents.
     */
    getLeafTexts(selector) {
      const root = document.querySelector(selector);
      if (!root) return [];
      const leaves = [];
      const walk = (node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          const trimmed = node.textContent.trim();
          if (trimmed) leaves.push(trimmed);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
          const children = node.children;
          if (children.length === 0 && node.textContent.trim()) {
            leaves.push(node.textContent.trim());
          } else {
            for (const child of children) walk(child);
          }
        }
      };
      walk(root);
      return leaves;
    },

    /**
     * Get all clickable elements on the page.
     * Clickable = <a>, <button>, elements with onclick, role="button", or tabindex >= 0.
     * @returns {Array<Object>} List of clickable elements with tag, text, href, and bounding box.
     */
    getClickableElements() {
      function getImplicitRole(el) {
        const tag = el.tagName.toLowerCase();
        if (tag === "a" && el.hasAttribute("href")) return "link";
        if (tag === "button") return "button";
        if (tag === "input") {
          const type = el.getAttribute("type") || "text";
          if (["button", "submit", "reset"].includes(type)) return "button";
          if (["checkbox", "radio"].includes(type)) return type;
          return "textbox";
        }
        if (tag === "select") return "listbox";
        if (tag === "textarea") return "textbox";
        return null;
      }

      function getCssSelector(el) {
        if (!(el instanceof Element)) return null;
        const path = [];
        while (el && el.nodeType === Node.ELEMENT_NODE) {
          let selector = el.nodeName.toLowerCase();
          if (el.id) {
            selector += `#${el.id}`;
            path.unshift(selector);
            break;
          } else {
            let sib = el,
              nth = 1;
            while ((sib = sib.previousElementSibling)) {
              if (sib.nodeName.toLowerCase() === selector) nth++;
            }
            if (nth > 1) selector += `:nth-of-type(${nth})`;
          }
          path.unshift(selector);
          el = el.parentElement;
        }
        return path.join(" > ");
      }

      const clickableSelectors = [
        "a[href]",
        "button",
        "[onclick]",
        '[role="button"]',
        "[tabindex]:not([tabindex='-1'])",
      ];
      const elements = document.querySelectorAll(clickableSelectors.join(","));
      const results = [];
      elements.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return; // skip invisible
        const info = {
          tag: el.tagName.toLowerCase(),
          text: el.innerText?.trim().slice(0, 100) || "",
          href: el.getAttribute("href") || null,
          role: el.getAttribute("role") || getImplicitRole(el),
          css_selector: getCssSelector(el),
          bbox: {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
          },
        };
        results.push(info);
      });
      return results;
    },

    /**
     * Example function for testing Playwright evaluate.
     * @param {string} name
     * @returns {string}
     */
    myInjectedFunction(name) {
      return `Hello from injected JS, ${name}!`;
    },
  };

  console.log("✅ utils.js injected successfully");
})();
