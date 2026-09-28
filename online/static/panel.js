/* Панель управления проектом: навигация, справка «?», разделы. */
'use strict';

const $ = id => document.getElementById(id);
const HORIZON = 20, HISTORY = 600;
const state = {
  targets: [], tgtIdx: 0,
  actual: [], markers: [], preds: null, predTick: null, predCount: 0,
  ws: null, u: null, prChart: null,
};
window.state = state; // отладка/тесты: доступ к графику из консоли

// ------------------------------------------------------------------ СПРАВКА
const tooltip = $('tooltip'), modalBack = $('modal-back'), modal = $('modal');
let pinnedKey = null;

function helpText(key, full) {
  const h = HELP[key];
  if (!h) return null;
  return full
    ? `<h3>${h.t}</h3><div class="what"><span class="lbl">Что произойдёт:</span> ${h.s}</div>` +
      `<div class="why"><span class="lbl">Подробнее и зачем:</span> ${h.d}</div>`
    : `<b>${h.t}</b><br>${h.s}`;
}

function initHelp() {
  // знак «?» уже в разметке; вешаем обработчики
  document.querySelectorAll('.help-btn').forEach(btn => {
    const key = btn.dataset.help;
    btn.addEventListener('mouseenter', e => {
      if (pinnedKey) return;
      const txt = helpText(key, false);
      if (!txt) return;
      tooltip.innerHTML = txt;
      tooltip.style.display = 'block';
      const r = btn.getBoundingClientRect();
      tooltip.style.left = Math.min(window.innerWidth - 360, r.left) + 'px';
      tooltip.style.top = (r.bottom + 6) + 'px';
    });
    btn.addEventListener('mouseleave', () => { if (!pinnedKey) tooltip.style.display = 'none'; });
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const h = HELP[key];
      if (!h) return;
      pinnedKey = key;
      tooltip.style.display = 'none';
      modal.innerHTML = helpText(key, true) +
        '<div style="margin-top:10px;text-align:right"><button onclick="this.closest(\'#modal-back\').style.display=\'none\';window.__pinned=null;">Закрыть</button></div>';
      modalBack.style.display = 'flex';
      window.__pinned = key;
    });
  });
  modalBack.addEventListener('click', e => {
    if (e.target === modalBack) { modalBack.style.display = 'none'; pinnedKey = null; }
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { modalBack.style.display = 'none'; pinnedKey = null; }
  });
}
window.__pinned = null;

// ------------------------------------------------------------------ НАВИГАЦИЯ
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.page').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('page-' + t.dataset.page).classList.add('active');
  if (t.dataset.page === 'online' && !state.u) makeUPlot();
}));

function showBanner(text, ok = true) {
  const b = $('banner');
  b.textContent = text;
  b.style.background = ok ? '#12321f' : '#4d1f24';
  b.style.color = ok ? '#9fe8bb' : '#ffb7c0';
  b.style.display = text ? 'block' : 'none';
}

// ------------------------------------------------------------------ ЗАГРУЗКА СПИСКОВ
async function jsonGet(url) { return (await fetch(url)).json(); }

async function loadModels(selIds) {
  const models = await jsonGet('/api/models');
  for (const sid of selIds) {
    const sel = $(sid);
    if (!sel) continue;
    const keep = sel.value;
    sel.innerHTML = sel.querySelector('option[value=""]')?.outerHTML ||
      (sel.id === 'tr-init' ? '<option value="">— с нуля —</option>' : '');
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.run_id;
      o.textContent = `${m.run_id} [${m.status}] ${m.mae != null ? 'MAE ' + m.mae.toFixed(2) : ''} — ${m.hypothesis || ''}`;
      sel.appendChild(o);
    }
    if (keep) sel.value = keep;
  }
  return models;
}

async function loadCsvs(selIds) {
  const csvs = await jsonGet('/api/datasets');
  for (const sid of selIds) {
    const sel = $(sid);
    sel.innerHTML = '';
    for (const c of csvs) {
      const o = document.createElement('option');
      o.value = o.textContent = c;
      sel.appendChild(o);
    }
    const yd = csvs.find(c => c.startsWith('your_data'));
    if (yd) sel.value = yd;
  }
}

// ------------------------------------------------------------------ ОБЗОР
async function refreshOverview() {
  const cfg = await jsonGet('/api/config');
  $('ov-profile').textContent = cfg.profile;
  $('ov-config').innerHTML = `
    <table>
      <tr><th>Параметр</th><th>Значение</th></tr>
      <tr><td>Окно истории</td><td>${cfg.sequence_length} шагов (${(cfg.sequence_length / 60).toFixed(1)} мин)</td></tr>
      <tr><td>Горизонт прогноза</td><td>${cfg.prediction_horizon} шагов (${cfg.prediction_horizon} с)</td></tr>
      <tr><td>Признаков / целей</td><td>${cfg.n_features} / ${cfg.n_targets}</td></tr>
      <tr><td>Цели</td><td>${cfg.targets.join(', ')}</td></tr>
      <tr><td>LR / эпохи / батч</td><td>${cfg.learning_rate} / ${cfg.num_epochs} / ${cfg.batch_size}</td></tr>
    </table>`;
  const models = await jsonGet('/api/models');
  $('ov-models').textContent = models.length;
  const csvs = await jsonGet('/api/datasets');
  $('ov-data').textContent = csvs.length;
  const sess = await jsonGet('/api/sessions');
  $('ov-sessions').textContent = sess.length;
}

// ------------------------------------------------------------------ ОБУЧЕНИЕ
$('btn-train')?.addEventListener('click', async () => {
  const body = {
    profile: $('tr-profile').value,
    csv: $('tr-csv').value,
    skip_rows: parseInt($('tr-skiprows').value || '0', 10),
    train_segments: parseInt($('tr-segments').value || '9', 10),
    max_epochs: parseInt($('tr-epochs').value || '100', 10),
    seed: parseInt($('tr-seed').value || '42', 10),
    notes: $('tr-notes').value || 'panel train',
    init_from: $('tr-init').value || null,
  };
  if ($('tr-lr').value) body.lr = parseFloat($('tr-lr').value);
  const r = await fetch('/api/train', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Ошибка запуска', false); return; }
  showBanner('Обучение запущено, задача ' + j.job_id);
  followJob(j.job_id, 'tr-log');
});

function followJob(jobId, logElId) {
  const el = $(logElId);
  el.style.display = 'block';
  const iv = setInterval(async () => {
    const j = await jsonGet('/api/jobs/' + jobId);
    el.textContent = (j.log || []).join('\n');
    el.scrollTop = el.scrollHeight;
    if (j.status !== 'running') {
      clearInterval(iv);
      showBanner(`Задача ${j.kind} завершена: ${j.status} (код ${j.returncode})`,
                 j.status === 'done');
      refreshOverview(); refreshRegistry();
    }
  }, 1500);
}

// ------------------------------------------------------------------ ТЕСТ-ПРОГНОЗ
$('btn-predict')?.addEventListener('click', async () => {
  const body = {
    run_id: $('pr-model').value, csv: $('pr-csv').value,
    start_row: parseInt($('pr-start').value || '0', 10),
  };
  showBanner('Считаю прогноз…');
  const r = await fetch('/api/predict', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Ошибка', false); return; }
  showBanner('');
  state.targets = j.targets;
  const tsel = $('pr-target');
  if (!tsel.options.length) {
    j.targets.forEach((t, i) => {
      const o = document.createElement('option'); o.value = o.textContent = t;
      tsel.appendChild(o);
    });
    const roll = [...tsel.options].find(o => o.text.startsWith('Roll'));
    if (roll) tsel.value = roll.text;
    tsel.addEventListener('change', () => drawPredict(j));
  }
  drawPredict(j);
  const rows = j.targets.map((t, i) =>
    `<tr><td>${t}</td><td>${j.mae ? j.mae[t].toFixed(4) : '—'}</td></tr>`).join('');
  $('pr-mae').innerHTML = `<table><tr><th>Цель</th><th>MAE (20 шагов)</th></tr>${rows}</table>`;
});

function drawPredict(j) {
  const tgt = $('pr-target').value || j.targets[0];
  const ti = j.targets.indexOf(tgt);
  const hist = j.history.map(r => r[ti]);
  const xHist = j.history.map((_, i) => i - j.history.length);
  const xPred = j.history.length + j.predictions.map((_, i) => i);
  const ticks = [...xHist, ...xPred];
  const sHist = [...hist, ...j.predictions.map(() => null)];
  const sPred = [...hist.map(() => null), ...j.predictions.map(r => r[ti])];
  let sAct;
  if (j.actual) {
    sAct = [...hist.map(() => null), ...j.actual.map(r => r[ti])];
  }
  const series = [
    { label: 'tick' },
    { label: 'прогноз', stroke: '#f5a623', width: 2, dash: [6, 4] },
    { label: 'факт', stroke: '#3aa2ff', width: 1.5, points: { show: true, size: 3 } },
  ];
  const data = [ticks, sPred, sHist];
  if (j.actual) { series.push({ label: 'факт (шаги)', stroke: '#35c777', width: 1.5 }); data.push(sAct); }
  if (!state.prChart) {
    const prCursor = makeCursorOpts();
    state.prChart = new uPlot({
      width: $('pr-chart').clientWidth - 8, height: 260,
      scales: { x: { time: false } }, series,
      cursor: prCursor.cursor,
      hooks: prCursor.hooks,
    }, data, $('pr-chart'));
    state.prChart.__readout = $('pr-cursor');
    attachZoomReset(state.prChart, $('btn-pr-reset'));
  } else {
    while (state.prChart.series.length < series.length)
      state.prChart.addSeries(series[state.prChart.series.length]);
    state.prChart.setData(data);
  }
}

// ------------------------------------------------------------------ ОНЛАЙН
// ------------------------------------------------------------------ ГРАФИКИ: общий зум/crosshair
function makeCursorOpts() {
  return {
    cursor: {
      drag: { x: true, y: true, uni: 5 },   // выделение рамкой = зум
      points: { show: (self, si) => si > 0 },
      focus: { alpha: 0.25 },
    },
    // координаты курсора: tick + значения всех линий в точке
    // (хуки uPlot — только в верхнем уровне opts при создании)
    hooks: {
      setCursor: [self => {
        const el = self.__readout;
        if (!el) return;
        const i = self.cursor.idx;
        if (i == null || self.data[0] == null || self.data[0][i] == null) {
          el.textContent = '';
          return;
        }
        const parts = [`tick ${self.data[0][i]}`];
        for (let s = 1; s < self.data.length; s++) {
          const v = self.data[s] ? self.data[s][i] : null;
          if (v == null) continue;
          parts.push(`${self.series[s].label}: ${(+v).toFixed(3)}`);
        }
        el.textContent = parts.join('   ');
      }],
    },
  };
}

function attachZoomReset(u, btnEl) {
  // setScale(null) в uPlot 1.6 не сбрасывает масштаб —
  // сброс = явный возврат к полному диапазону данных
  const reset = () => {
    const x = u.data[0];
    if (x && x.length) u.setScale('x', { min: x[0], max: x[x.length - 1] });
  };
  btnEl?.addEventListener('click', reset);
  u.over.addEventListener('dblclick', reset);   // двойной клик — тоже сброс
}

function makeUPlot() {
  const wrap = $('chart-wrap');
  const cursor = makeCursorOpts();
  const opts = {
    width: Math.max(10, wrap.clientWidth - 8), height: 300,
    scales: { x: { time: false } },
    cursor: cursor.cursor,
    hooks: cursor.hooks,
    series: [
      { label: 'tick' },
      { label: 'прогноз', stroke: '#f5a623', width: 2, dash: [6, 4] },
      { label: 'факт', stroke: '#3aa2ff', width: 1.5, points: { show: true, size: 3 } },
      { label: 'input_anom', points: { show: true, size: 7 }, stroke: '#ef5466', width: 1 },
      { label: 'error_anom', points: { show: true, size: 7 }, stroke: '#b45cff', width: 1 },
    ],
  };
  state.u = new uPlot(opts, [[], [], [], [], []], wrap);
  state.u.__readout = $('on-cursor');
  attachZoomReset(state.u, $('btn-on-reset'));
}

function drawOnline() {
  if (!state.u) return;
  const lastTick = state.actual.length ? state.actual[state.actual.length - 1].tick : 0;
  const windowStart = Math.max(0, lastTick - HISTORY);
  const actMap = new Map(), predMap = new Map(), markMap = new Map();
  for (const p of state.actual) if (p.tick >= windowStart) actMap.set(p.tick, p.val);
  if (state.preds && state.predTick !== null)
    state.preds.forEach((v, i) => predMap.set(state.predTick + 1 + i, v));
  for (const m of state.markers) markMap.set(m.tick, m);
  const ticks = [...new Set([...actMap.keys(), ...predMap.keys()])].sort((a, b) => a - b);
  const a = [], p = [], mInp = [], mErr = [];
  for (const t of ticks) {
    const av = actMap.get(t) ?? null, pv = predMap.get(t) ?? null;
    a.push(av); p.push(pv);
    const mk = markMap.get(t);
    mInp.push(mk && mk.inp ? (av ?? pv) : null);
    mErr.push(mk && mk.err ? (av ?? pv) : null);
  }
  state.u.setData([ticks, p, a, mInp, mErr]);
}

function buildTargetBar() {
  const bar = $('tgt-bar'); bar.innerHTML = '';
  state.targets.forEach((t, i) => {
    const el = document.createElement('span');
    el.className = 'tgt' + (i === state.tgtIdx ? ' active' : '');
    el.style.cssText = 'border:1px solid var(--border);border-radius:12px;padding:2px 10px;font-size:12px;cursor:pointer;color:var(--dim)';
    el.textContent = t;
    el.onclick = () => { state.tgtIdx = i; state.actual = []; buildTargetBar(); };
    bar.appendChild(el);
  });
}

$('btn-start')?.addEventListener('click', async () => {
  const mode = $('sel-mode').value;
  const body = {
    model: $('sel-model').value, csv: $('sel-csv').value,
    speed: mode === 'playback' ? parseFloat($('sel-speed').value) : 1.0,
    time_scale: mode === 'live' ? parseFloat($('sel-speed').value) : 1.0,
    start_row: parseInt($('start-row').value || '0', 10),
    limit: mode === 'playback' && $('limit').value ? parseInt($('limit').value, 10) : null,
    source: mode,
    seed: parseInt($('seed').value || '42', 10),
    duration_s: $('duration').value ? parseFloat($('duration').value) : null,
    anomaly_every: $('anomevery').value ? parseInt($('anomevery').value, 10) : null,
  };
  const r = await fetch('/api/session/start', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Не удалось запустить', false); return; }
  state.predCount = 0; state.actual = []; state.markers = []; state.preds = null;
  if (!state.u) makeUPlot();
  connectWs();
});

$('btn-stop')?.addEventListener('click', () => fetch('/api/session/stop', { method: 'POST' }));
$('btn-pause')?.addEventListener('click', async () => {
  const btn = $('btn-pause');
  const paused = btn.dataset.paused !== 'true';
  await fetch('/api/session/pause', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ paused }) });
  btn.dataset.paused = String(paused);
});

function connectWs() {
  if (state.ws) state.ws.close();
  state.ws = new WebSocket(`ws://${location.host}/ws`);
  state.ws.onopen = () => { $('btn-start').disabled = true; $('btn-stop').disabled = false; $('btn-pause').disabled = false; showBanner(''); };
  state.ws.onclose = () => { $('btn-start').disabled = false; $('btn-stop').disabled = true; $('btn-pause').disabled = true; };
  state.ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'tick') handleTick(m);
    if (m.type === 'end') { showBanner(`Сессия завершена. tick: ${m.status ? m.status.tick : '—'}`); }
  };
}

function handleTick(m) {
  if (!state.targets.length && m.actual) {
    state.targets = Object.keys(m.actual);
    state.tgtIdx = Math.max(0, state.targets.indexOf('Roll(градусы)'));
    buildTargetBar();
  }
  const tgt = state.targets[state.tgtIdx];
  if (tgt && m.actual) {
    state.actual.push({ tick: m.tick, val: m.actual[tgt] });
    if (state.actual.length > HISTORY + HORIZON) state.actual.shift();
  }
  state.markers.push({ tick: m.tick, inp: m.input_anomaly, err: m.error_anomaly });
  if (state.markers.length > HISTORY + HORIZON) state.markers.shift();
  if (m.predicted && m.preds) {
    state.preds = m.preds.map(row => row[state.tgtIdx]);
    state.predTick = m.tick;
    state.predCount += 1;
  }
  drawOnline();
  $('st-tick').textContent = m.tick;
  $('st-pred').textContent = state.predCount;
  $('st-ms').textContent = m.inference_ms ? m.inference_ms.toFixed(0) + ' мс' : '—';
  $('st-ratio').textContent = m.error_ratio == null ? '—' : m.error_ratio.toFixed(2);
  $('st-inp').className = 'marker ' + (m.input_anomaly ? 'on' : 'off');
  $('st-err').className = 'marker ' + (m.error_anomaly ? 'on' : 'off');
}

// ------------------------------------------------------------------ ДАННЫЕ
async function refreshData() {
  const csvs = await jsonGet('/api/datasets');
  $('dt-list').innerHTML = '<table><tr><th>Файл</th></tr>' +
    csvs.map(c => `<tr><td>${c}</td></tr>`).join('') + '</table>';
  const sess = await jsonGet('/api/sessions');
  $('dt-sessions').innerHTML = sess.length
    ? '<table><tr><th>Сессия</th></tr>' + sess.map(s =>
        `<tr><td>${s.name}</td></tr>`).join('') + '</table>'
    : '<span style="color:var(--dim)">пока нет — запустите режим B в разделе «Онлайн»</span>';
  const fs = $('ft-session');
  const keep = fs.value;
  fs.innerHTML = sess.map(s => `<option value="${s.name}">${s.name}</option>`).join('');
  if (keep) fs.value = keep;
}

$('btn-upload')?.addEventListener('click', async () => {
  const f = $('dt-file').files[0];
  if (!f) { showBanner('Выберите файл', false); return; }
  const text = await f.text();
  const r = await fetch('/api/datasets/upload', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ name: f.name, data_b64: btoa(unescape(encodeURIComponent(text))) }) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Ошибка загрузки', false); return; }
  showBanner(`Загружено: ${j.file} (${j.rows} строк, ${j.columns} колонок)`);
  refreshData(); loadCsvs(['tr-csv', 'sel-csv', 'pr-csv']);
});

$('btn-syn')?.addEventListener('click', async () => {
  const r = await fetch('/api/synthetic', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ rows: parseInt($('sy-rows').value || '30000', 10),
                           seed: parseInt($('sy-seed').value || '42', 10) }) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Ошибка', false); return; }
  showBanner('Генерация запущена, задача ' + j.job_id);
  followJob(j.job_id, 'jb-log');
});

// ------------------------------------------------------------------ ДООБУЧЕНИЕ
$('btn-ft')?.addEventListener('click', async () => {
  const body = {
    session: $('ft-session').value, run_id: $('ft-base').value,
    all_rows: $('ft-allrows').checked, skip_cooldown: $('ft-skipcd').checked,
  };
  if ($('ft-minrows').value) body.min_rows = parseInt($('ft-minrows').value, 10);
  if ($('ft-epochs').value) body.max_epochs = parseInt($('ft-epochs').value, 10);
  const r = await fetch('/api/finetune', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { showBanner(j.error || 'Ошибка', false); return; }
  showBanner('Дообучение запущено, задача ' + j.job_id);
  followJob(j.job_id, 'ft-log');
});

// ------------------------------------------------------------------ РЕЕСТР
let regSelected = null;
async function refreshRegistry() {
  const models = await jsonGet('/api/models');
  $('rg-table').innerHTML = '<table><tr><th></th><th>run_id</th><th>статус</th>' +
    '<th>MAE</th><th>заметка</th></tr>' + models.map(m => `
    <tr style="cursor:pointer" data-run="${m.run_id}"
        class="${regSelected === m.run_id ? 'sel' : ''}">
      <td>${regSelected === m.run_id ? '▶' : ''}</td>
      <td>${m.run_id}</td>
      <td class="${m.status === 'production' ? 'status-ok' : m.status && m.status.startsWith('rejected') ? 'status-err' : ''}">${m.status || ''}</td>
      <td>${m.mae != null ? m.mae.toFixed(3) : '—'}</td>
      <td>${m.hypothesis || ''}</td></tr>`).join('') + '</table>';
  $('rg-table').querySelectorAll('tr[data-run]').forEach(tr =>
    tr.addEventListener('click', () => { regSelected = tr.dataset.run; refreshRegistry(); showManifest(regSelected); }));
}

async function showManifest(runId) {
  const pre = $('rg-manifest');
  pre.style.display = 'block';
  pre.textContent = 'загрузка…';
  try {
    const m = await jsonGet('/api/models').then(list => list.find(x => x.run_id === runId));
    pre.textContent = `run_id: ${runId}\nstatus: ${m.status}\nMAE: ${m.mae ?? '—'}\n` +
      `заметка: ${m.hypothesis}\nокно: ${m.sequence_length} → горизонт: ${m.prediction_horizon}`;
  } catch { pre.textContent = 'нет данных'; }
}

$('btn-promote')?.addEventListener('click', async () => {
  if (!regSelected) { showBanner('Выберите модель в таблице', false); return; }
  const r = await fetch('/api/models/promote', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ run_id: regSelected }) });
  const j = await r.json();
  showBanner(r.ok ? `${j.run_id} → production` : j.error, r.ok);
  refreshRegistry();
});

$('btn-delete')?.addEventListener('click', async () => {
  if (!regSelected) { showBanner('Выберите модель в таблице', false); return; }
  if (!confirm(`Удалить ${regSelected} необратимо?`)) return;
  const r = await fetch('/api/models/delete', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ run_id: regSelected, confirm: true }) });
  const j = await r.json();
  showBanner(r.ok ? 'Удалено' : j.error, r.ok);
  regSelected = null;
  refreshRegistry(); refreshOverview();
});

// ------------------------------------------------------------------ ЗАДАЧИ
async function refreshJobs() {
  const jobs = await jsonGet('/api/jobs');
  if (!jobs.length) { $('jb-list').innerHTML = '<span style="color:var(--dim)">задач нет</span>'; return; }
  $('jb-list').innerHTML = '<table><tr><th>id</th><th>тип</th><th>статус</th><th>старт</th></tr>' +
    jobs.map(j => `<tr><td>${j.id}</td><td>${j.kind}</td>` +
      `<td class="${j.status === 'done' ? 'status-ok' : j.status === 'running' ? 'status-run' : 'status-err'}">${j.status}</td>` +
      `<td>${j.started}</td></tr>`).join('') + '</table>';
}

// ------------------------------------------------------------------ INIT
async function init() {
  initHelp();
  const cfg = await jsonGet('/api/config');
  const tsel = $('tr-profile');
  cfg.profiles.forEach(p => {
    const o = document.createElement('option'); o.value = o.textContent = p;
    tsel.appendChild(o);
  });
  tsel.value = cfg.profile;
  await Promise.all([loadModels(['sel-model', 'pr-model', 'ft-base', 'tr-init']),
                     loadCsvs(['tr-csv', 'sel-csv', 'pr-csv']),
                     refreshOverview(), refreshData(), refreshRegistry(), refreshJobs()]);
  // график онлайна создаётся лениво — при первом открытии вкладки «Онлайн»
  const st = await jsonGet('/api/session/status');
  if (st.running) { document.querySelector('.tab[data-page="online"]').click(); makeUPlot(); connectWs(); }
  setInterval(() => { refreshJobs(); }, 5000);
}
init();
