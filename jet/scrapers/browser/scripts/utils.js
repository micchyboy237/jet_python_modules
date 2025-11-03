(() => {
  // Avoid overwriting if already loaded
  if (window.__utilsInjected__) return;
  window.__utilsInjected__ = true;

  // --- INIT SCRIPT: Track JS click listeners ---
  (() => {
    if (window.__clickTrackerInjected__) return;
    window.__clickTrackerInjected__ = true;

    window.__clickableElements = new Set();
    const origAddEventListener = EventTarget.prototype.addEventListener;

    EventTarget.prototype.addEventListener = function (
      type,
      listener,
      options
    ) {
      if (type === "click" && this instanceof Element) {
        window.__clickableElements.add(this);
      }
      return origAddEventListener.call(this, type, listener, options);
    };

    console.log("✅ Click-tracker initialized");
  })();

  // --- UTILS NAMESPACE ---
  window.Utils = {
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

    scrollIntoView(selector) {
      const el = document.querySelector(selector);
      if (!el) return false;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      return true;
    },

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
        if (rect.width === 0 && rect.height === 0) return;
        results.push({
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
        });
      });
      return results;
    },

    /** NEW: Get elements with JavaScript click listeners. */
    getJSClickableElements() {
      if (!window.__clickableElements) return [];
      const elements = Array.from(window.__clickableElements);
      const getCssSelector = (el) => {
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
      };

      return elements.map((el) => ({
        tag: el.tagName.toLowerCase(),
        text: el.innerText?.trim().slice(0, 100) || "",
        hasHref: !!el.getAttribute("href"),
        css_selector: getCssSelector(el),
      }));
    },

    myInjectedFunction(name) {
      return `Hello from injected JS, ${name}!`;
    },
  };

  console.log("✅ utils.js injected successfully");
})();
