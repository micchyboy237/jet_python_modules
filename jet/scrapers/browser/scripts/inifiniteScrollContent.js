/**
 * Advanced content stabilization + infinite scroll handler
 * - Supports lazy loading, virtualized lists, and delayed APIs
 * - Combines height, DOM mutation, and bottom detection
 */

async function stabilizeContent(options = {}) {
  const {
    maxWaitMs = 12000,
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

    if (heightChanged || contentChanged || domChanged) {
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

// --- Usage Example ---

(async () => {
  const controller = new AbortController();

  // Optional: cancel after 15s hard timeout
  setTimeout(() => controller.abort(), 15000);

  const result = await stabilizeContent({
    maxWaitMs: 12000,
    contentSelector: null, // e.g. ".feed-item"
    signal: controller.signal,
    debug: true,
  });

  console.log("Result:", result);
})();
