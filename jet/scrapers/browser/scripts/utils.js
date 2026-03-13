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
      options,
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
          const type = (el.getAttribute("type") || "text").toLowerCase();
          if (["button", "submit", "reset", "image"].includes(type))
            return "button";
          if (["checkbox", "radio"].includes(type)) return type;
          if (["hidden", "file"].includes(type)) return null; // usually skip
          return "textbox"; // text, email, password, tel, url, search, number, date etc.
        }

        if (tag === "textarea") return "textbox";
        if (tag === "select") return "listbox";
        if (tag === "option") return null; // usually not directly clickable

        // ARIA buttons / other interactive roles
        const role = el.getAttribute("role")?.toLowerCase();
        if (
          role === "button" ||
          role === "link" ||
          role === "checkbox" ||
          role === "radio" ||
          role === "tab" ||
          role === "menuitem"
        ) {
          return role;
        }

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

      // ──────────────────────────────────────────────
      //  Broader selector: interactive + form controls
      // ──────────────────────────────────────────────
      const interactiveSelectors = [
        "a[href]",
        "button",
        "input",
        "select",
        "textarea",
        "[role='button']",
        "[role='link']",
        "[role='checkbox']",
        "[role='radio']",
        "[role='tab']",
        "[role='menuitem']",
        "[onclick]",
        "[tabindex]:not([tabindex='-1'])",
      ];

      const elements = document.querySelectorAll(
        interactiveSelectors.join(","),
      );

      const results = [];

      elements.forEach((el) => {
        // Skip elements that are practically invisible / non-interactive
        const rect = el.getBoundingClientRect();
        if (rect.width <= 1 || rect.height <= 1) return; // tightened a bit

        // Also skip hidden inputs & elements with visibility/display none
        const style = window.getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden") return;

        const role = getImplicitRole(el);
        if (!role) return; // skip elements we don't consider interactive

        // For form fields → try to get associated label text
        let labelText = "";
        if (["textbox", "checkbox", "radio", "listbox"].includes(role)) {
          const id = el.id;
          if (id) {
            const label = document.querySelector(`label[for="${id}"]`);
            if (label) labelText = label.innerText.trim();
          }
          // fallback: aria-label or placeholder
          labelText = labelText || el.getAttribute("aria-label") || "";
          labelText = labelText || el.placeholder?.trim() || "";
        }

        results.push({
          tag: el.tagName.toLowerCase(),
          type:
            el.tagName.toLowerCase() === "input"
              ? el.type || "text"
              : undefined,
          text: (el.innerText || labelText || el.value || el.placeholder || "")
            .trim()
            .slice(0, 120),
          href: el.getAttribute("href") || null,
          role: el.getAttribute("role") || role,
          id: el.id || null,
          name: el.name || null,
          css_selector: getCssSelector(el),
          bbox: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
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
