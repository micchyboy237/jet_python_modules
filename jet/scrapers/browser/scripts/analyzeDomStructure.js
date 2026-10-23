(async () => {
  console.log("=== 🧠 Final Optimized LLM Context Generator ===");
  console.time("Generation");

  // 1. Helper: Check if string is pure alphabet
  const isPureAlpha = (str) => str && /^[a-zA-Z]+$/.test(str);

  // 2. Helper: Generate stable selector for LLM usage
  function getStableSelector(el) {
    if (el.id && isPureAlpha(el.id)) return `#${el.id}`;
    const pureClasses = Array.from(el.classList).filter(isPureAlpha);
    if (pureClasses.length > 0)
      return pureClasses
        .slice(0, 2)
        .map((c) => `.${c}`)
        .join("");
    return null;
  }

  // 3. Helper: Get clean, meaningful text (filter out noise)
  function getCleanText(el) {
    let text = el.textContent || "";
    text = text.replace(/\s+/g, " ").trim();

    // Filter out noise: JSON, CSS, or very short fragments
    if (text.length < 3) return null;
    if (
      text.startsWith("{") ||
      text.startsWith("[") ||
      text.startsWith(".") ||
      text.startsWith("#")
    )
      return null;
    if (text.includes("stylesheet") || text.includes("function(")) return null;

    return text.substring(0, 60);
  }

  // 4. Helper: Create a "signature" to detect repetitive siblings
  function getSignature(el) {
    const tag = el.tagName.toLowerCase();
    const id = el.id && isPureAlpha(el.id) ? `#${el.id}` : "";
    const pureClasses = Array.from(el.classList)
      .filter(isPureAlpha)
      .sort()
      .join(".");
    return `${tag}${id}${pureClasses ? "." + pureClasses : ""}`;
  }

  // 5. Recursive Builder with Cleaning & Collapsing
  function buildOptimizedTree(element, depth = 0) {
    const indent = "  ".repeat(depth);
    const tag = element.tagName.toLowerCase();

    // Skip non-visual tags
    if (["script", "style", "link", "meta", "noscript", "iframe"].includes(tag))
      return "";

    const id = element.id;
    const classes = Array.from(element.classList);
    const pureAlphaClasses = classes.filter(isPureAlpha);
    const selector = getStableSelector(element);

    // Decide if this element is "significant"
    const isSignificant =
      isPureAlpha(id) ||
      pureAlphaClasses.length > 0 ||
      [
        "nav",
        "main",
        "header",
        "footer",
        "section",
        "article",
        "form",
        "table",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "button",
        "a",
        "input",
      ].includes(tag);

    const cleanText = getCleanText(element);
    const children = Array.from(element.children);

    // --- REPETITION DETECTION LOGIC ---
    if (children.length > 2) {
      const firstSig = getSignature(children[0]);
      const allSame = children.every(
        (child) => getSignature(child) === firstSig,
      );

      if (allSame) {
        // It's a repetitive list! Collapse it.
        let result = "";
        if (isSignificant || selector || cleanText) {
          let attrs = "";
          if (id && isPureAlpha(id)) attrs += ` id="${id}"`;
          if (pureAlphaClasses.length > 0)
            attrs += ` class="${pureAlphaClasses.join(" ")}"`;
          if (selector) attrs += ` selector="${selector}"`;

          result += `${indent}<${tag}${attrs}> [List of ${children.length} items]\n`;

          // Show ONE example child
          const exampleChild = children[0];
          const childTag = exampleChild.tagName.toLowerCase();
          const childId =
            exampleChild.id && isPureAlpha(exampleChild.id)
              ? ` id="${exampleChild.id}"`
              : "";
          const childPureClasses = Array.from(exampleChild.classList).filter(
            isPureAlpha,
          );
          const childClassAttr =
            childPureClasses.length > 0
              ? ` class="${childPureClasses.join(" ")}"`
              : "";
          const childSelector = getStableSelector(exampleChild);
          const childSelectorAttr = childSelector
            ? ` selector="${childSelector}"`
            : "";

          const childText = getCleanText(exampleChild);
          const textStr = childText ? ` "${childText}"` : "";

          result += `${indent}  └─ <${childTag}${childId}${childClassAttr}${childSelectorAttr}>${textStr} ...\n`;
        } else {
          // If parent isn't significant, just recurse into one child to keep structure
          result += buildOptimizedTree(children[0], depth);
        }
        return result;
      }
    }
    // --- END REPETITION DETECTION ---

    // RECURSIVE CHECK: Do any children have content?
    let hasSignificantChildren = false;
    let childrenOutput = "";

    for (const child of children) {
      const childResult = buildOptimizedTree(child, depth + 1);
      if (childResult) {
        hasSignificantChildren = true;
        childrenOutput += childResult;
      }
    }

    // CLEANING RULE: If an element has no text, no selector, and no significant children, skip it.
    if (!isSignificant && !cleanText && !hasSignificantChildren && !selector) {
      return "";
    }

    // Build the line
    let attrs = "";
    if (id && isPureAlpha(id)) attrs += ` id="${id}"`;
    if (pureAlphaClasses.length > 0)
      attrs += ` class="${pureAlphaClasses.join(" ")}"`;
    if (selector) attrs += ` selector="${selector}"`;

    let result = `${indent}<${tag}${attrs}>`;

    // Add text if present
    if (cleanText) {
      result += ` "${cleanText}"`;
    }

    result += "\n";

    // Add children if they exist
    if (childrenOutput) {
      result += childrenOutput;
    }

    return result;
  }

  // 6. Execute on the main body
  const skeleton = buildOptimizedTree(document.body);

  console.log("\n--- 📋 Copy this into your LLM Prompt ---\n");
  console.log(skeleton);
  console.log("\n--- End of Context ---\n");

  // 7. Auto-copy for convenience
  try {
    await navigator.clipboard.writeText(skeleton);
    console.log("✨ Skeleton copied to clipboard!");
  } catch (e) {
    if (typeof copy !== "undefined") {
      copy(skeleton);
      console.log("✨ Skeleton copied via devtools helper!");
    }
  }

  console.timeEnd("Generation");
})();
