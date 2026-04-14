/**
 * Advanced content stabilization + infinite scroll + video extraction
 * - Supports lazy loading, virtualized lists, and delayed APIs
 * - Now extracts and prints video data after each scroll (up to 5 scrolls)
 * - Combines height, DOM mutation, and bottom detection
 */

// === DATA EXTRACTION HELPERS (provided by user) ===
function findVideoWithPreviewContainer(element) {
  if (!element || element === document.body) return null;
  const hasVideo = element.querySelector("video") !== null;
  const hasImg = element.querySelector("img") !== null;
  if (hasVideo && hasImg) {
    return element;
  }
  return findVideoWithPreviewContainer(element.parentElement);
}

function getSrcOrDataSrc(element) {
  if (!element) return null;
  const src = element.getAttribute("src")?.trim();
  const dataSrc = element.getAttribute("data-src")?.trim();
  return src || dataSrc || null;
}

function extractData() {
  const anchors = document.querySelectorAll(".text-secondary");
  const data = Array.from(anchors)
    .map((a) => {
      let url = a.href?.trim() || "";
      const text = a.textContent?.trim() || "";
      const hashIndex = url.indexOf("#");
      if (hashIndex !== -1) {
        url = url.substring(0, hashIndex);
      }
      if (!url || !text) {
        return null;
      }
      const container = findVideoWithPreviewContainer(a);
      if (!container) {
        return {
          url,
          text,
          thumbnail: null,
          preview: null,
        };
      }
      const img = container.querySelector("img");
      const thumbnail = img ? getSrcOrDataSrc(img) : null;
      const video = container.querySelector("video");
      let preview = null;
      if (video) {
        preview =
          getSrcOrDataSrc(video) ||
          video.querySelector("source")?.getAttribute("src")?.trim() ||
          null;
      }
      return {
        url,
        text,
        thumbnail,
        preview,
      };
    })
    .filter((item) => item !== null);
  return data;
}

// === PRINT HELPER ===
function printExtractedData(data, scrollNumber) {
  console.log(
    `\n📊 === Scroll ${scrollNumber} - Extracted ${data.length} video items ===`,
  );
  data.forEach((item, index) => {
    console.log(`  ${index + 1}. "${item.text}"`);
    console.log(`     🔗 ${item.url}`);
    if (item.thumbnail) console.log(`     🖼️  Thumbnail: ${item.thumbnail}`);
    if (item.preview) console.log(`     🎥 Preview:   ${item.preview}`);
    console.log(""); // empty line between items
  });
  console.log(`✅ Total so far this scroll: ${data.length} items\n`);
}

/**
 * Advanced content stabilization + infinite scroll handler
 * - Supports lazy loading, virtualized lists, and delayed APIs
 * - Combines height, DOM mutation, and bottom detection
 */
async function stabilizeContent(options = {}) {
  const {
    maxWaitMs = 12000,
    onContentChange = null, // New: callback when content change is detected
    checkIntervalMs = 400,
    bottomDelayMs = 2500,
    maxNoChangeStreak = 3,
    scrollStepRatio = 0.8,
    contentSelector = null, // e.g. ".item" (optional but recommended)
    signal = null, // AbortController.signal
    debug = true,
  } = options;

  const root = document.scrollingElement || document.documentElement;

  const log = (...args) => {
    if (debug) console.log(...args);
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const isAborted = () => signal?.aborted;

  const getScrollHeight = () => root.scrollHeight;

  const getScrollTop = () => root.scrollTop;

  const getViewportHeight = () => window.innerHeight;

  const isAtBottom = () =>
    getScrollTop() + getViewportHeight() >= getScrollHeight() - 2;

  const scrollStep = () => {
    const delta = getViewportHeight() * scrollStepRatio;
    root.scrollBy(0, delta);
  };

  const scrollToBottom = () => {
    root.scrollTo(0, getScrollHeight());
  };

  const getContentCount = () => {
    if (!contentSelector) return null;
    return document.querySelectorAll(contentSelector).length;
  };

  // Mutation tracking
  let mutationCount = 0;
  const observer = new MutationObserver((mutations) => {
    mutationCount += mutations.length;
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  const startTime = Date.now();

  let lastHeight = getScrollHeight();
  let lastContentCount = getContentCount();
  let noChangeStreak = 0;

  log("🚀 Starting advanced stabilization...");

  // --- Phase 0: Initial kickstart scroll ---
  scrollStep();
  await sleep(800);

  // --- Phase 1: Progressive scrolling + stabilization ---
  while (Date.now() - startTime < maxWaitMs) {
    if (isAborted()) break;

    await sleep(checkIntervalMs);

    const currentHeight = getScrollHeight();
    const currentContentCount = getContentCount();
    const currentMutations = mutationCount;

    const heightChanged = currentHeight > lastHeight;
    const contentChanged =
      currentContentCount !== null && currentContentCount !== lastContentCount;
    const domChanged = currentMutations > 0;

    const anyChange = heightChanged || contentChanged || domChanged;

    if (anyChange && typeof onContentChange === "function") {
      onContentChange("Phase 1 - Content change detected during stabilization");
    }

    if (anyChange) {
      log("📈 Content change detected", {
        heightDelta: currentHeight - lastHeight,
        contentDelta:
          currentContentCount !== null
            ? currentContentCount - lastContentCount
            : null,
        mutations: currentMutations,
      });

      lastHeight = currentHeight;
      lastContentCount = currentContentCount;
      mutationCount = 0;
      noChangeStreak = 0;

      scrollStep();
      await sleep(800);
      continue;
    }

    noChangeStreak++;

    log(`⏳ No change streak: ${noChangeStreak}`);

    if (noChangeStreak >= maxNoChangeStreak) {
      log("✅ Initial stabilization reached");
      break;
    }
  }

  // --- Phase 2: Bottom re-check loop ---
  let bottomChecks = 0;

  while (Date.now() - startTime < maxWaitMs) {
    if (isAborted()) break;

    if (!isAtBottom()) {
      scrollToBottom();
      await sleep(1000);
    }

    await sleep(bottomDelayMs);

    const currentHeight = getScrollHeight();
    const currentContentCount = getContentCount();

    const heightChanged = currentHeight > lastHeight;
    const contentChanged =
      currentContentCount !== null && currentContentCount !== lastContentCount;

    if (
      (heightChanged || contentChanged) &&
      typeof onContentChange === "function"
    ) {
      onContentChange("Phase 2 - New content after bottom wait");
    }

    if (heightChanged || contentChanged) {
      log("📈 New content after bottom wait", {
        heightDelta: currentHeight - lastHeight,
        contentDelta:
          currentContentCount !== null
            ? currentContentCount - lastContentCount
            : null,
      });

      lastHeight = currentHeight;
      lastContentCount = currentContentCount;
      noChangeStreak = 0;
      bottomChecks++;

      scrollStep();
      await sleep(1000);
      continue;
    }

    noChangeStreak++;
    log(`⏳ Bottom no-change streak: ${noChangeStreak}`);

    if (noChangeStreak >= maxNoChangeStreak) {
      log("🏁 Final stabilization complete");
      break;
    }
  }

  observer.disconnect();

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  log(`✅ Done in ${elapsed}s`, {
    bottomChecks,
  });

  return {
    success: true,
    elapsedMs: Date.now() - startTime,
    bottomChecks,
  };
}

// === PRINT HELPER (improved to show accumulated) ===
function printExtractedData(
  data,
  scrollNumber,
  isAccumulated = false,
  changeReason = "",
) {
  const title = isAccumulated
    ? `📊 === Accumulated Results (${data.length} unique videos) ===`
    : `📊 === Scroll ${scrollNumber} - Extracted ${data.length} video items ===`;

  console.log(`\n${title}`);
  if (changeReason) console.log(`   Trigger: ${changeReason}`);

  data.forEach((item, index) => {
    console.log(`  ${index + 1}. "${item.text}"`);
    console.log(`     🔗 ${item.url}`);
    if (item.thumbnail) console.log(`     🖼️  Thumbnail: ${item.thumbnail}`);
    if (item.preview) console.log(`     🎥 Preview:   ${item.preview}`);
    console.log("");
  });
  console.log(`✅ Total: ${data.length} items\n`);
}

/**
 * Main function: Scroll up to 5 times and extract videos after each scroll
 * Now also logs accumulated results on every content change detected inside stabilization
 */
async function scrollAndExtractVideos(maxScrolls = 5) {
  console.log(
    "🚀 Starting video extraction with infinite scroll (max 5 scrolls)...",
  );

  // Accumulator: deduplicated by URL
  const accumulated = new Map(); // url → item

  const onContentChange = (reason) => {
    const freshData = extractData();
    let newCount = 0;

    freshData.forEach((item) => {
      if (!accumulated.has(item.url)) {
        accumulated.set(item.url, item);
        newCount++;
      }
    });

    const currentAccumulated = Array.from(accumulated.values());
    console.log(`\n🔄 Content change detected! +${newCount} new videos`);
    printExtractedData(currentAccumulated, null, true, reason);
  };

  let totalItems = 0;
  let previousCount = 0;

  for (let scrollNum = 1; scrollNum <= maxScrolls; scrollNum++) {
    console.log(`\n🔄 === Performing Scroll ${scrollNum}/${maxScrolls} ===`);

    // 1. Stabilize the page + trigger onContentChange for every detected change
    await stabilizeContent({
      maxWaitMs: 10000,
      onContentChange: onContentChange, // ← New
      contentSelector: null,
      debug: true,
    });

    // 2. Final extraction after full stabilization (for per-scroll summary)
    const extracted = extractData();

    // Add to accumulator (in case some were missed)
    extracted.forEach((item) => {
      if (!accumulated.has(item.url)) {
        accumulated.set(item.url, item);
      }
    });

    // 3. Print per-scroll summary
    const finalThisScroll = Array.from(accumulated.values());
    printExtractedData(finalThisScroll, scrollNum);

    totalItems = accumulated.size;

    // Optional: stop early if no new items
    if (extracted.length === previousCount && scrollNum > 1) {
      console.log("🛑 No new videos found. Stopping early.");
      break;
    }

    previousCount = extracted.length;

    // Small extra scroll to trigger next batch
    if (scrollNum < maxScrolls) {
      window.scrollBy(0, window.innerHeight * 0.6);
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  const finalAccumulated = Array.from(accumulated.values());
  console.log(`\n🎉 Finished all scrolls!`);
  console.log(
    `📈 Total unique video items extracted: ${finalAccumulated.length}`,
  );
  return {
    totalUniqueItems: finalAccumulated.length,
    scrollsPerformed:
      typeof scrollNum === "undefined" ? maxScrolls : scrollNum - 1,
  };
}

// --- Usage Example ---
(async () => {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 60000);

  await scrollAndExtractVideos(5);
})();
