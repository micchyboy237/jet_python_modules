(function () {
  console.clear();
  console.log("🕸️ Starting Shadow DOM Pierce & Cloudflare Detection...");

  const CF_KEYWORDS = [
    "challenge",
    "turnstile",
    "cloudflare",
    "verify",
    "human",
  ];
  const foundElements = [];

  // --- Helper: Check if element is relevant to Cloudflare ---
  function isCloudflareRelated(el) {
    if (!el || !el.tagName) return false;

    const src = el.src || "";
    const id = el.id || "";
    const className = el.className || "";
    const text = el.innerText || "";

    // Check for Cloudflare iframe sources
    if (src.includes("challenges.cloudflare.com") || src.includes("turnstile"))
      return true;

    // Check for common CF classes/IDs
    if (
      id.includes("cf-") ||
      className.includes("cf-") ||
      className.includes("turnstile")
    )
      return true;

    // Check text content for keywords
    if (text.toLowerCase().includes("verify you are human")) return true;

    return false;
  }

  // --- Helper: Highlight Element ---
  function highlightElement(el, labelText) {
    // Avoid double-highlighting
    if (el.dataset.cfHighlighted) return;
    el.dataset.cfHighlighted = "true";

    const originalOutline = el.style.outline;
    el.style.outline = "4px solid #ff0055";
    el.style.outlineOffset = "-4px";
    el.style.position = "relative"; // Ensure z-index works

    const label = document.createElement("div");
    label.textContent = labelText;
    label.style.position = "absolute";
    label.style.top = "-25px";
    label.style.left = "0";
    label.style.background = "#ff0055";
    label.style.color = "white";
    label.style.padding = "2px 6px";
    label.style.fontSize = "11px";
    label.style.fontWeight = "bold";
    label.style.borderRadius = "4px";
    label.style.zIndex = "999999";
    label.style.pointerEvents = "none";
    label.style.whiteSpace = "nowrap";

    el.appendChild(label);
  }

  // --- Recursive Shadow DOM Traversal ---
  function traverseShadowDOM(root, path = []) {
    // Get all elements in current root
    const elements = root.querySelectorAll("*");

    elements.forEach((el) => {
      // 1. Check if this element is a Cloudflare target
      if (isCloudflareRelated(el)) {
        const currentPath = [...path, el.tagName.toLowerCase()];
        if (el.id) currentPath.push(`#${el.id}`);

        console.warn(`🎯 Found CF Element: ${currentPath.join(" > ")}`);

        // If it's an iframe, we can't click it directly from parent, but we can highlight it
        let label = "CF Shadow Element";
        if (el.tagName === "IFRAME") label = "CF Iframe (Shadow)";
        else if (el.className.includes("turnstile")) label = "Turnstile Widget";

        highlightElement(el, label);
        foundElements.push({
          element: el,
          path: currentPath.join(" > "),
          type: el.tagName,
        });
      }

      // 2. Pierce Shadow Root if it exists
      if (el.shadowRoot) {
        traverseShadowDOM(el.shadowRoot, [...path, `${el.tagName} (Shadow)`]);
      }
    });
  }

  // --- Start Traversal from Document Body ---
  traverseShadowDOM(document.body);

  // --- Also check for top-level iframes just in case ---
  const topIframes = document.querySelectorAll(
    'iframe[src*="challenges.cloudflare.com"]',
  );
  topIframes.forEach((iframe) => {
    if (!iframe.dataset.cfHighlighted) {
      console.warn("🎯 Found Top-Level CF Iframe");
      highlightElement(iframe, "CF Iframe (Top)");
      foundElements.push({
        element: iframe,
        path: "body > iframe",
        type: "IFRAME",
      });
    }
  });

  // --- Results ---
  if (foundElements.length > 0) {
    console.group(
      `✅ Found ${foundElements.length} Cloudflare-related elements`,
    );
    foundElements.forEach((item, index) => {
      console.log(`${index + 1}. Type: ${item.type} | Path: ${item.path}`);
    });
    console.groupEnd();

    console.info(
      "💡 Elements are highlighted in pink. Note: If it's an iframe, you must switch context to interact with it.",
    );
  } else {
    console.log("❌ No Cloudflare challenges detected in DOM or Shadow DOM.");
  }
})();
