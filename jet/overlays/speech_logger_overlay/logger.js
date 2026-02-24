window.__LOGGER_READY__ = false;

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("log-container");
  const search = document.getElementById("search");
  const buttons = document.querySelectorAll("#filters button");

  let activeLevels = new Set([...buttons].map((b) => b.dataset.level));

  buttons.forEach((btn) => {
    btn.classList.add("active");
    btn.onclick = () => {
      btn.classList.toggle("active");
      const lvl = btn.dataset.level;
      btn.classList.contains("active")
        ? activeLevels.add(lvl)
        : activeLevels.delete(lvl);
      applyFilters();
    };
  });

  search.oninput = applyFilters;

  window.addLog = function ({ level, message, timestamp, render_mode }) {
    const el = document.createElement("div");
    el.className = `log ${level}`;
    el.dataset.level = level;

    const ts = document.createElement("span");
    ts.className = "timestamp";
    ts.textContent = timestamp;

    el.appendChild(ts);
    el.appendChild(document.createTextNode(" "));

    if (render_mode === "html") {
      const wrapper = document.createElement("div");
      wrapper.innerHTML = message;
      el.appendChild(wrapper);
    } else {
      const text = document.createElement("span");
      text.textContent = message;
      el.appendChild(text);
    }

    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
    applyFilters();
  };

  function applyFilters() {
    const q = search.value.toLowerCase();
    [...container.children].forEach((log) => {
      const matchLevel = activeLevels.has(log.dataset.level);
      const matchText = log.textContent.toLowerCase().includes(q);
      log.style.display = matchLevel && matchText ? "" : "none";
    });
  }

  // 🔔 Signal Python that JS is fully ready
  window.__LOGGER_READY__ = true;
});
