/**
 * Live threshold slider for truth-check.html.
 *
 * Loads site/data/comparison.json (the per-channel share gaps) and
 * site/data/simulations.json (the 300 simulation predictions).
 * On every slider input event:
 *   - Re-classifies each channel as OVER / UNDER / ACCURATE based on
 *     the chosen threshold.
 *   - Re-renders the per-channel table (label cells, sorted by abs gap).
 *   - Recomputes accuracy + macro-F1 across the 300 predictions and
 *     updates the metric tiles.
 */
(function () {
  'use strict';

  const slider = document.getElementById('threshold-slider');
  const valueEl = document.getElementById('threshold-value');
  const tbody = document.getElementById('channels-tbody');
  const countOver = document.getElementById('count-over');
  const countUnder = document.getElementById('count-under');
  const countAccurate = document.getElementById('count-accurate');
  const accuracyEl = document.getElementById('metric-accuracy');
  const macroEl = document.getElementById('metric-macro-f1');

  if (!slider || !tbody) return;

  let comparisonData = null;
  let simulationData = null;

  function classify(gapPp, threshold) {
    if (gapPp > threshold) return 'OVER_CREDITED';
    if (gapPp < -threshold) return 'UNDER_CREDITED';
    return 'ACCURATE';
  }

  function labelClass(label) {
    if (label === 'OVER_CREDITED') return 'label-over';
    if (label === 'UNDER_CREDITED') return 'label-under';
    return 'label-accurate';
  }

  function labelDisplay(label) {
    return label.replace('_', ' ').replace('CREDITED', 'credited');
  }

  function renderChannels(threshold) {
    if (!comparisonData) return;
    const sorted = [...comparisonData.channels].sort(
      (a, b) => Math.abs(b.share_gap_pp) - Math.abs(a.share_gap_pp)
    );
    let nOver = 0, nUnder = 0, nAccurate = 0;
    const rows = sorted.map(c => {
      const label = classify(c.share_gap_pp, threshold);
      if (label === 'OVER_CREDITED') nOver++;
      else if (label === 'UNDER_CREDITED') nUnder++;
      else nAccurate++;
      const gap = c.share_gap_pp;
      const sign = gap >= 0 ? '+' : '';
      return `
        <tr>
          <td>${c.channel}</td>
          <td class="num">${c.model_share_pct.toFixed(1)}%</td>
          <td class="num">${c.measured_share_pct.toFixed(1)}%</td>
          <td class="num">${sign}${gap.toFixed(1)}pp</td>
          <td><span class="label ${labelClass(label)}">${labelDisplay(label)}</span></td>
        </tr>`;
    }).join('');
    tbody.innerHTML = rows;
    if (countOver) countOver.textContent = nOver;
    if (countUnder) countUnder.textContent = nUnder;
    if (countAccurate) countAccurate.textContent = nAccurate;
  }

  function computeMetrics(threshold) {
    if (!simulationData) return;
    const labels = ['OVER_CREDITED', 'UNDER_CREDITED', 'ACCURATE'];
    const conf = {};
    labels.forEach(t => { conf[t] = {}; labels.forEach(p => { conf[t][p] = 0; }); });
    let correct = 0;
    for (const s of simulationData) {
      const pred = classify(s.share_gap_pp, threshold);
      conf[s.true_label][pred]++;
      if (s.true_label === pred) correct++;
    }
    const total = simulationData.length;
    const accuracy = correct / total;
    let macroF1 = 0;
    for (const cls of labels) {
      const tp = conf[cls][cls];
      let fp = 0, fn = 0;
      for (const other of labels) {
        if (other !== cls) {
          fp += conf[other][cls];
          fn += conf[cls][other];
        }
      }
      const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
      macroF1 += f1;
    }
    macroF1 /= labels.length;
    if (accuracyEl) accuracyEl.textContent = accuracy.toFixed(3);
    if (macroEl) macroEl.textContent = macroF1.toFixed(3);
  }

  function update() {
    const t = parseFloat(slider.value);
    if (valueEl) valueEl.textContent = t.toFixed(1) + 'pp';
    renderChannels(t);
    computeMetrics(t);
  }

  Promise.all([
    fetch('data/comparison.json').then(r => r.json()),
    fetch('data/simulations.json').then(r => r.json()),
  ]).then(([comp, sims]) => {
    comparisonData = comp;
    simulationData = sims;
    update();
    slider.addEventListener('input', update);
  }).catch(err => {
    console.error('Failed to load data:', err);
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="5">Could not load comparison data. Try refreshing the page.</td></tr>';
    }
  });
})();
