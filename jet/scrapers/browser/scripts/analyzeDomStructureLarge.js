(async () => {
  console.log("=== 🧠 Cleaned LLM Context Generator ===");
  console.time("Generation");

  // 1. Helper: Check if string is pure alphabet
  const isPureAlpha = (str) => str && /^[a-zA-Z]+$/.test(str);

  // 2. Helper: Generate stable selector
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

  // 3. Helper: Get clean, meaningful text
  function getCleanText(el) {
    let text = el.textContent || "";
    // Remove excessive whitespace
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

  // 4. Recursive Builder with Cleaning Logic
  function buildCleanTree(element, depth = 0) {
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

    // RECURSIVE CHECK: Do any children have content?
    let hasSignificantChildren = false;
    let childrenOutput = "";

    for (const child of children) {
      const childResult = buildCleanTree(child, depth + 1);
      if (childResult) {
        hasSignificantChildren = true;
        childrenOutput += childResult;
      }
    }

    // CLEANING RULE: If an element has no text, no selector, and no significant children, skip it.
    if (!isSignificant && !cleanText && !hasSignificantChildren) {
      return "";
    }

    // Build the line
    let attrs = "";
    if (id && isPureAlpha(id)) attrs += ` id="${id}"`;
    if (pureAlphaClasses.length > 0)
      attrs += ` class="${pureAlphaClasses.join(" ")}"`;
    if (selector) attrs += ` selector="${selector}"`; // <-- KEY ADDITION FOR LLM

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

  // 5. Execute
  const skeleton = buildCleanTree(document.body);

  console.log("\n--- 📋 Copy this into your LLM Prompt ---\n");
  console.log(skeleton);
  console.log("\n--- End of Context ---\n");

  // 6. Auto-copy
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
