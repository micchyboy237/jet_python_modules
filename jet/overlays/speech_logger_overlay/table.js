// table.js

let segments = [];
let initialized = false;

function buildRow(seg) {
  return `
    <tr data-id="${seg.num}">
      <td>${seg.num}</td>
      <td>${seg.start}</td>
      <td>${seg.end}</td>
      <td>${(seg.end - seg.start).toFixed(1)}</td>
      <td>${seg.prob.toFixed(3)}</td>
      <td>${seg.frame_start}</td>
      <td>${seg.frame_end}</td>
      <td class="type-${seg.type}">${seg.type}</td>
    </tr>
  `;
}

function renderTable() {
  const tbody = document.querySelector("#segments-table-wrapper tbody");
  if (!tbody) return;
  tbody.innerHTML = segments.map(buildRow).join("");
}

/* ========= PUBLIC API ========= */

window.SegmentsTable = {
  init() {
    if (initialized) return;

    const container = document.getElementById("log-container");
    const wrapper = document.createElement("div");
    wrapper.id = "segments-table-wrapper";

    wrapper.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Start</th>
            <th>End</th>
            <th>Duration</th>
            <th>Prob</th>
            <th>Frame Start</th>
            <th>Frame End</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    `;

    container.appendChild(wrapper);
    initialized = true;
  },

  setAll(data) {
    segments = [...data];
    renderTable();
  },

  add(segment) {
    segments.push(segment);
    renderTable();
  },

  update(segment) {
    segments = segments.map((s) => (s.num === segment.num ? segment : s));
    renderTable();
  },

  delete(segmentId) {
    segments = segments.filter((s) => s.num !== segmentId);
    renderTable();
  },
};
