function checkEventListeners(selector) {
  const config = {
    color: "#FF0000",
    width: 3,
    highlightDuration: 2000, // milliseconds to keep highlight visible
  };

  function getActiveEvents(el) {
    const events = [];

    // Check inline event handlers
    if (el.hasAttribute("onclick")) events.push("onclick (inline)");
    if (el.hasAttribute("onmousedown")) events.push("onmousedown (inline)");
    if (el.hasAttribute("onmouseup")) events.push("onmouseup (inline)");
    if (el.hasAttribute("onkeydown")) events.push("onkeydown (inline)");
    if (el.hasAttribute("ontouchstart")) events.push("ontouchstart (inline)");

    // Check JS-added event listeners using Chrome DevTools API
    try {
      if (typeof getEventListeners === "function") {
        const listeners = getEventListeners(el);
        if (listeners) {
          Object.keys(listeners).forEach((eventName) => {
            const listenerCount = listeners[eventName].length;
            if (listenerCount > 0) {
              events.push(`${eventName} (${listenerCount})`);
            }
          });
        }
      }
    } catch (e) {
      console.warn("getEventListeners not available:", e.message);
    }

    return events;
  }

  function getAllEventTypes(el) {
    const allEvents = [];

    try {
      if (typeof getEventListeners === "function") {
        const listeners = getEventListeners(el);
        if (listeners) {
          Object.keys(listeners).forEach((eventName) => {
            const listenerDetails = listeners[eventName];
            listenerDetails.forEach((listener, index) => {
              allEvents.push({
                type: eventName,
                index: index,
                listener: listener.listener,
                useCapture: listener.useCapture || false,
                once: listener.once || false,
                passive: listener.passive || false,
              });
            });
          });
        }
      }
    } catch (e) {}

    return allEvents;
  }

  function processElement(el) {
    const events = getActiveEvents(el);
    const allEventDetails = getAllEventTypes(el);

    // Build selector string for logging
    let sel = el.tagName.toLowerCase();
    if (el.id) {
      sel += `#${el.id}`;
    } else if (el.className && typeof el.className === "string") {
      const firstClass = el.className.split(" ")[0];
      if (firstClass) sel += `.${firstClass}`;
    }

    // Highlight the element temporarily
    const originalOutline = el.style.outline;
    const originalOutlineOffset = el.style.outlineOffset;
    el.style.outline = `${config.width}px solid ${config.color}`;
    el.style.outlineOffset = "2px";

    // Remove highlight after duration
    setTimeout(() => {
      el.style.outline = originalOutline;
      el.style.outlineOffset = originalOutlineOffset;
    }, config.highlightDuration);

    // Log results
    if (events.length > 0) {
      console.group(
        `%c[JS Events Found] %c${sel}`,
        "color: #ff00ff; font-weight: bold;",
        "color: #fff;",
      );
      console.log(`%cEvents: ${events.join(", ")}`, "color: #ffa500;");

      if (allEventDetails.length > 0) {
        console.log(
          "%cDetailed Listener Info:",
          "color: #00ffff; font-weight: bold;",
        );
        allEventDetails.forEach((detail) => {
          console.log(`  • ${detail.type}:`, detail.listener);
          console.log(
            `    - useCapture: ${detail.useCapture}, once: ${detail.once}, passive: ${detail.passive}`,
          );
        });
      }

      console.log("%cDOM Element:", "color: #00ff00; font-weight: bold;");
      console.log(el);
      console.groupEnd();
    } else {
      console.log(
        `%c[No JS Events] %c${sel}`,
        "color: #888888; font-weight: bold;",
        "color: #fff;",
      );
    }

    console.log("---");
  }

  // Validate selector
  if (!selector || typeof selector !== "string") {
    console.error(
      "%cError: Please provide a valid CSS selector string",
      "color: red; font-weight: bold;",
    );
    return null;
  }

  // Query elements
  let elements;
  try {
    elements = document.querySelectorAll(selector);
  } catch (e) {
    console.error(
      `%cError: Invalid CSS selector "${selector}"`,
      "color: red; font-weight: bold;",
    );
    console.error(e.message);
    return null;
  }

  if (elements.length === 0) {
    console.warn(
      `%cNo elements found for selector: "${selector}"`,
      "color: orange; font-weight: bold;",
    );
    return [];
  }

  console.log(
    `%cFound ${elements.length} element(s) for selector: "${selector}"`,
    "color: cyan; font-weight: bold; font-size: 14px;",
  );

  const results = [];
  elements.forEach((el, index) => {
    const events = getActiveEvents(el);
    results.push({
      element: el,
      index: index,
      hasEvents: events.length > 0,
      events: events,
      allEventDetails: getAllEventTypes(el),
    });
    processElement(el);
  });

  // Summary
  const withEvents = results.filter((r) => r.hasEvents).length;
  const withoutEvents = results.length - withEvents;

  console.log(
    `%cSummary:`,
    "color: yellow; font-weight: bold; font-size: 13px;",
  );
  console.log(`  Total elements: ${results.length}`);
  console.log(`  With JS events: ${withEvents}`);
  console.log(`  Without JS events: ${withoutEvents}`);

  return results;
}

// checkEventListeners('a[data-jk]');
// checkEventListeners('#job_b8cc43c29a4ae42f');
