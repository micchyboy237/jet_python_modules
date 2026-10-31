(() => {
  const config = { color: "#FF0000", width: 3 };

  function getActiveEvents(el) {
    const events = [];
    if (el.hasAttribute("onclick")) events.push("onclick (inline)");
    try {
      if (typeof getEventListeners === "function") {
        const listeners = getEventListeners(el);
        if (listeners) {
          Object.keys(listeners).forEach((eventName) => {
            if (
              [
                "click",
                "mousedown",
                "mouseup",
                "touchstart",
                "keydown",
              ].includes(eventName)
            ) {
              events.push(`${eventName} (${listeners[eventName].length})`);
            }
          });
        }
      }
    } catch (e) {}
    return events;
  }

  function isInteractive(el) {
    if (
      ["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "SUMMARY"].includes(
        el.tagName,
      )
    )
      return true;
    const role = el.getAttribute("role");
    if (role && ["button", "link", "tab", "menuitem"].includes(role))
      return true;
    if (el.hasAttribute("tabindex") && el.getAttribute("tabindex") !== "-1")
      return true;
    if (el.hasAttribute("onclick")) return true;
    try {
      if (typeof getEventListeners === "function") {
        const l = getEventListeners(el);
        if (l && (l.click || l.mousedown || l.touchstart)) return true;
      }
    } catch (e) {}
    return false;
  }

  function processElement(el) {
    if (el.dataset.highlighted === "true") return;
    if (!isInteractive(el)) return;

    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") return;

    // Highlight
    el.style.outline = `${config.width}px solid ${config.color}`;
    el.style.outlineOffset = "2px";
    el.dataset.highlighted = "true";

    // Get Info
    const events = getActiveEvents(el);
    let sel = el.tagName.toLowerCase();
    if (el.id) sel += `#${el.id}`;
    else if (el.className) sel += `.${el.className.split(" ")[0]}`;

    // Print Logs
    if (events.length > 0) {
      console.log(
        `%c[JS Event] %c${sel}`,
        "color: #ff00ff; font-weight: bold;",
        "color: #fff;",
      );
      console.log(`   Events: ${events.join(", ")}`);
    } else {
      console.log(
        `%c[Standard] %c${sel}`,
        "color: #00ff00; font-weight: bold;",
        "color: #fff;",
      );
    }

    // Print the actual DOM element for inspection
    console.log(el);
    console.log("---");
  }

  // Recursive scan of current DOM
  document.querySelectorAll("*").forEach(processElement);

  // Watch for future recursive additions
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) {
          processElement(node);
          node.querySelectorAll("*").forEach(processElement);
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
  console.log(
    "%cRecursive scan active. Click elements below to inspect.",
    "color: cyan; font-weight: bold;",
  );
})();
