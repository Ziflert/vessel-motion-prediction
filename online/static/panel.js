/* Панель управления проектом: навигация, справка «?», разделы. */
'use strict';

const $ = id => document.getElementById(id);
const HORIZON = 20, HISTORY = 600;
const state = {
  targets: [], tgtIdx: 0,
  actual: [], markers: [], preds: null, predTick: null, predCount: 0,
  forecasts: [],  // история прогнозов для серии «прогноз был» ({tick, vals})
  rows: [],       // сырые строки записи для плавающих показателей (F1, bounded)
  live: [],       // выбранные параметры карточек-показателей
  uncertain: { mean: null, std: null, alert: false },  // N4/C4: лента MC-Dropout
  ws: null, u: null, prChart: null,
  // выбор линий графика: несколько параметров + скрытие/цвета (запрос заказчика)
  vwSel: [], vwColors: {}, vwHidden: {},
  prColors: {}, prHidden: {}, prResult: null,
  onColors: {},
  // онлайн: дополнительные линии данных (выбор чипами, из сырых строк записи)
  onExtra: [], onExtraColors: {}, onExtraHidden: {},
  // графики скрытых вкладок после смены темы помечаются устаревшими —
  // пересоздаются при открытии вкладки (иначе строятся шириной 10px)
  uStale: false, prStale: false,
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
  if (t.dataset.page === 'online') {
    if (!state.u) makeUPlot();
    else if (state.uStale) {   // график был пересоздан в скрытой вкладке (10px)
      state.u.destroy(); state.u = null; makeUPlot(); drawOnline();
      state.uStale = false;
    }
    renderLiveCards();
    refreshLiveSelect();   // селект карточек-показателей (F1)
    refreshOnLines();      // чипы дополнительных линий данных
  }
  if (t.dataset.page === 'predict' && state.prStale && state.prResult) {
    drawPredict(state.prResult);   // пересоздание после смены темы в скрытой вкладке
    state.prStale = false;
  }
}));

function showBanner(text, ok = true) {
  const b = $('banner');
  b.textContent = text;
  b.style.background = ok ? '#12321f' : '#4d1f24';
  b.style.color = ok ? '#9fe8bb' : '#ffb7c0';
  b.style.display = text ? 'block' : 'none';
}

// ------------------------------------------------- ДНЕВНОЙ/НОЧНОЙ РЕЖИМ
// Переключатель 🌙/☀: класс body.light меняет CSS-переменные; выбор
// сохраняется в localStorage. Графики пересоздаются — у них цвета осей
// берутся из переменных темы (иначе оси нечитаемы в дневном режиме).
function initTheme() {
  if (localStorage.getItem('panel-theme') === 'light')
    document.body.classList.add('light');
  updateThemeBtn();
}

function updateThemeBtn() {
  const b = $('btn-theme');
  if (b) b.textContent = document.body.classList.contains('light') ? '☀' : '🌙';
}

function isLight() { return document.body.classList.contains('light'); }

$('btn-theme')?.addEventListener('click', () => {
  document.body.classList.toggle('light');
  localStorage.setItem('panel-theme', isLight() ? 'light' : 'dark');
  updateThemeBtn();
  redrawAllCharts();
});

// ------------------------------------------------- ЧИПЫ ЛИНИЙ ГРАФИКА
// У каждой линии: кружок цвета (input color) + название; клик по названию —
// скрыть/показать линию. Скрытое состояние сохраняется между перерисовками.
const VW_PALETTE = ['#3aa2ff', '#35c777', '#f5a623', '#e05bc4',
                    '#b45cff', '#ef5466', '#2fd6c8', '#ffd166'];

// input type=color принимает только #rrggbb — конвертируем rgba/#rgb
function toHex6(color) {
  if (typeof color !== 'string') return '#3aa2ff';
  if (color.startsWith('#')) {
    if (color.length === 7) return color;
    if (color.length === 4)
      return '#' + [...color.slice(1)].map(c => c + c).join('');
    return '#3aa2ff';
  }
  const m = color.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const parts = m[1].split(',').map(s => parseFloat(s));
    if (parts.length >= 3)
      return '#' + parts.slice(0, 3).map(v =>
        Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
  }
  return '#3aa2ff';
}

function buildChips(containerId, itemsGetter) {
  const wrap = $(containerId);
  if (!wrap) return;
  const items = itemsGetter();
  wrap.innerHTML = '';
  for (const it of items) {
    const chip = document.createElement('span');
    chip.className = 'ser-chip' + (it.show ? '' : ' off');
    chip.title = 'клик по названию — скрыть/показать линию';
    const cinp = document.createElement('input');
    cinp.type = 'color';
    cinp.value = it.color;
    cinp.title = 'цвет линии';
    cinp.onclick = e => e.stopPropagation();
    cinp.oninput = () => it.onColor(cinp.value);
    const name = document.createElement('b');
    name.textContent = it.label;
    chip.appendChild(cinp);
    chip.appendChild(name);
    chip.onclick = () => it.onToggle();
    wrap.appendChild(chip);
  }
}

function serItems(u) {
  if (!u) return [];
  const items = [];
  for (let s = 1; s < u.series.length; s++) {
    const ser = u.series[s];
    if (!ser.label) continue;
    items.push({
      label: ser.label,
      // uPlot хранит цвет в _stroke (stroke после конструктора — функция-резолвер);
      // приоритет — выбор пользователя (persistMap, обновляется кружком цвета)
      color: toHex6((u.__persistMap && u.__persistMap[ser.label] != null)
        ? u.__persistMap[ser.label]
        : (typeof ser.stroke === 'function' ? ser._stroke : ser.stroke)),
      show: ser.show !== false,
      onToggle: () => {
        const next = !(ser.show !== false);
        u.setSeries(s, { show: next });
        if (u.__persistHidden) {
          if (next) delete u.__persistHidden[ser.label];
          else u.__persistHidden[ser.label] = true;
        }
        buildChips(u.__chipsId, () => serItems(u));
      },
      onColor: (c) => {
        u.setSeries(s, { stroke: c });
        if (u.__persistMap) u.__persistMap[ser.label] = c;
      },
    });
  }
  return items;
}

function bindChips(u, chipsId, persistMap) {
  u.__chipsId = chipsId;
  u.__persistMap = persistMap;
  u.__persistHidden = {};
  buildChips(chipsId, () => serItems(u));
}

// Цвета осей из CSS-переменных текущей темы
function axesFromTheme() {
  const cs = getComputedStyle(document.body);
  const stroke = (cs.getPropertyValue('--dim') || '').trim() || '#7b8ba1';
  const grid = (cs.getPropertyValue('--border') || '').trim() || '#263141';
  return [
    { stroke, grid: { stroke: grid }, ticks: { stroke: grid } },
    { stroke, grid: { stroke: grid }, ticks: { stroke: grid } },
  ];
}

// Смена темы: пересоздать графики ТОЛЬКО видимых вкладок; скрытые помечаются
// устаревшими и пересоздаются при открытии вкладки (иначе график строится
// шириной 10px и «сплющивается» — баг, замеченный заказчиком).
function redrawAllCharts() {
  if (state.prChart) {
    if ($('pr-chart').clientWidth >= 20) {
      const j = state.prResult;
      if (j) drawPredict(j);
      state.prStale = false;
    } else state.prStale = true;
  }
  if (state.u) {
    if ($('chart-wrap').clientWidth >= 20) {
      state.u.destroy();
      state.u = null;
      makeUPlot();
      drawOnline();
      state.uStale = false;
    } else state.uStale = true;
  }
  if (vwData && state.vwSel.length && $('vw-chart').clientWidth >= 20) drawView();
}

// ------------------------------------------------------------------ ЗАГРУЗКА СПИСКОВ
async function jsonGet(url) { return (await fetch(url)).json(); }

// Безопасный разбор ответа: при 500/обрыве сервер может вернуть не-JSON —
// тогда панели молча «не нажимались» (баннер висел вечно). Теперь всегда
// получаем объект с полем error/detail и показываем его пользователю.
async function jsonResp(r) {
  try {
    const j = await r.json();
    if (!r.ok && j && !j.error && j.detail) j.error =
      typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
    return j;
  } catch {
    return { error: `HTTP ${r.status} — сервер вернул не-JSON ответ (подробности в логе сервера)` };
  }
}

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
  const j = await jsonResp(r);
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
  const j = await jsonResp(r);
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

function prColor(label) {
  if (!state.prColors[label]) {
    const DEF = { 'прогноз': '#f5a623', 'факт': '#3aa2ff', 'факт (шаги)': '#35c777' };
    const used = Object.keys(state.prColors).length;
    state.prColors[label] = DEF[label] || VW_PALETTE[used % VW_PALETTE.length];
  }
  return state.prColors[label];
}

function drawPredict(j) {
  state.prResult = j;
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
  // серии: цвета из выбора панели (кружки в чипах), скрытие — по чипам;
  // подписи фиксированы на пересоздании (uPlot не обновляет легенду на лету)
  const series = [
    { label: 'tick' },
    { label: 'прогноз', stroke: prColor('прогноз'), width: 2, dash: [6, 4],
      show: !state.prHidden['прогноз'] },
    { label: 'факт', stroke: prColor('факт'), width: 1.5,
      points: { show: true, size: 3 }, show: !state.prHidden['факт'] },
  ];
  const data = [ticks, sPred, sHist];
  if (j.actual) {
    series.push({ label: 'факт (шаги)', stroke: prColor('факт (шаги)'), width: 1.5,
                  show: !state.prHidden['факт (шаги)'] });
    data.push(sAct);
  }
  // перенос скрытия/цветов из предыдущего графика (в т.ч. легендный toggle)
  if (state.prChart) {
    for (const ser of state.prChart.series) {
      if (!ser.label) continue;
      if (ser.show === false) state.prHidden[ser.label] = true;
      else delete state.prHidden[ser.label];
    }
    state.prChart.destroy();
    state.prChart = null;
  }
  const prCursor = makeCursorOpts();
  state.prChart = new uPlot({
    width: Math.max(10, $('pr-chart').clientWidth - 8), height: 260,
    scales: { x: { time: false } },
    axes: axesFromTheme(),
    series,
    cursor: prCursor.cursor,
    hooks: prCursor.hooks,
  }, data, $('pr-chart'));
  state.prChart.__readout = $('pr-cursor');
  attachZoomReset(state.prChart, $('btn-pr-reset'));
  bindChips(state.prChart, 'pr-chips', state.prColors);
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

function onColor(label) {
  if (!state.onColors[label]) {
    const DEF = {
      'прогноз (сейчас)': '#f5a623',
      'факт': '#3aa2ff',
      'прогноз был (1 шаг назад)': '#35c777',
      [`прогноз был (${HORIZON} шагов назад)`]: '#e05bc4',
      'input_anom': '#ef5466',
      'error_anom': '#b45cff',
      'лента −': 'rgba(245,166,35,.5)',
      'лента +': 'rgba(245,166,35,.5)',
    };
    const used = Object.keys(state.onColors).length;
    state.onColors[label] = DEF[label] || VW_PALETTE[used % VW_PALETTE.length];
  }
  return state.onColors[label];
}

function onExtraColor(col) {
  if (!state.onExtraColors[col]) {
    const used = Object.keys(state.onExtraColors).length;
    state.onExtraColors[col] = VW_PALETTE[(used + 1) % VW_PALETTE.length];
  }
  return state.onExtraColors[col];
}

function makeUPlot() {
  const wrap = $('chart-wrap');
  const cursor = makeCursorOpts();
  // стандартные серии (цель графика) + дополнительные линии данных (чипы);
  // индексы ленты [7, 8] фиксированы — доп. линии добавляются после них
  const series = [
    { label: 'tick' },
    { label: 'прогноз (сейчас)', stroke: onColor('прогноз (сейчас)'), width: 2, dash: [6, 4] },
    { label: 'факт', stroke: onColor('факт'), width: 1.5, points: { show: true, size: 3 } },
    { label: 'прогноз был (1 шаг назад)', stroke: onColor('прогноз был (1 шаг назад)'), width: 1.2 },
    { label: `прогноз был (${HORIZON} шагов назад)`, stroke: onColor(`прогноз был (${HORIZON} шагов назад)`), width: 1.2, dash: [2, 3] },
    { label: 'input_anom', points: { show: true, size: 7 }, stroke: onColor('input_anom'), width: 1 },
    { label: 'error_anom', points: { show: true, size: 7 }, stroke: onColor('error_anom'), width: 1 },
    // N4/C4: лента неопределённости MC-Dropout (mean ± z·std, шир. по выбранной цели)
    { label: 'лента −', stroke: onColor('лента −'), width: 1, dash: [2, 4] },
    { label: 'лента +', stroke: onColor('лента +'), width: 1, dash: [2, 4] },
  ];
  for (const col of state.onExtra) {
    series.push({ label: col, stroke: onExtraColor(col), width: 1.5,
                  show: !state.onExtraHidden[col] });
  }
  const opts = {
    width: Math.max(10, wrap.clientWidth - 8), height: 300,
    scales: { x: { time: false } },
    axes: axesFromTheme(),
    cursor: cursor.cursor,
    hooks: cursor.hooks,
    series,
    bands: [{ series: [7, 8], fill: 'rgba(245,166,35,.12)' }],
  };
  state.u = new uPlot(opts, Array.from({ length: series.length }, () => []), wrap);
  state.u.__readout = $('on-cursor');
  attachZoomReset(state.u, $('btn-on-reset'));
  bindChips(state.u, 'on-chips', state.onColors);
  window.state.u = state.u; // отладка/приёмка: доступ из консоли
}

// Пересоздание онлайнового графика при смене набора доп. линий;
// при скрытой вкладке — откладывается до её открытия (state.uStale)
function rebuildOnline() {
  if (!state.u) return;
  if ($('chart-wrap').clientWidth < 20) { state.uStale = true; return; }
  for (const ser of state.u.series) {
    if (!ser.label) continue;
    if (ser.show === false) state.onExtraHidden[ser.label] = true;
    else delete state.onExtraHidden[ser.label];
  }
  state.u.destroy();
  state.u = null;
  makeUPlot();
  drawOnline();
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
  // История прошедших прогнозов: для каждого момента T берём, что модель
  // предсказала для T lag'ами назад — 1 шаг и полная дистанция HORIZON.
  // Так видно наглядно (не по цифрам), насколько хорошо/плохо прогноз сбывается.
  const past1 = new Map(), pastH = new Map();
  for (const f of state.forecasts) {
    const t1 = f.tick + 1, v1 = f.vals ? f.vals[0] : null;
    if (v1 !== undefined && t1 >= windowStart) past1.set(t1, v1 ? v1[state.tgtIdx] : null);
    const tH = f.tick + HORIZON, vH = f.vals ? f.vals[HORIZON - 1] : null;
    if (vH !== undefined && tH >= windowStart) pastH.set(tH, vH ? vH[state.tgtIdx] : null);
  }
  const ticks = [...new Set([...actMap.keys(), ...predMap.keys(), ...past1.keys(), ...pastH.keys()])]
    .sort((a, b) => a - b);
  const a = [], p = [], p1 = [], pH = [], mInp = [], mErr = [], lo = [], hi = [];
  // N4/C4: лента неопределённости вокруг текущего прогноза (mean ± z·std,
  // шир. по выбранной цели; обновляется с cadence — лента персистентна)
  const z = 1.96, uc = state.uncertain;
  const uncLo = new Map(), uncHi = new Map();
  if (uc.mean && uc.std && state.predTick !== null)
    uc.mean.forEach((row, i) => {
      const t = state.predTick + 1 + i;
      if (t < windowStart) return;
      const m = row[state.tgtIdx], s = (uc.std[i] || [])[state.tgtIdx];
      if (m == null || s == null) return;
      uncLo.set(t, m - z * s); uncHi.set(t, m + z * s);
    });
  for (const t of ticks) {
    const av = actMap.get(t) ?? null, pv = predMap.get(t) ?? null;
    a.push(av); p.push(pv);
    p1.push(past1.get(t) ?? null); pH.push(pastH.get(t) ?? null);
    lo.push(uncLo.get(t) ?? null); hi.push(uncHi.get(t) ?? null);
    const mk = markMap.get(t);
    mInp.push(mk && mk.inp ? (av ?? pv) : null);
    mErr.push(mk && mk.err ? (av ?? pv) : null);
  }
  const data = [ticks, p, a, p1, pH, mInp, mErr, lo, hi];
  // дополнительные линии данных (выбор чипами) — из уже полученных сырых
  // строк записи: пересчитывать ничего не нужно, tick-сообщение содержит
  // полную строку (запрос заказчика: волна/скорость + качка на одном графике)
  for (const col of state.onExtra) {
    const vals = new Map();
    for (const r of state.rows) {
      if (r.__tick >= windowStart && r[col] != null) vals.set(r.__tick, r[col]);
    }
    data.push(ticks.map(t => vals.get(t) ?? null));
  }
  state.u.setData(data);
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
    speed: mode === 'playback' ? (parseFloat($('sel-speed').value) || 1.0) : 1.0,
    time_scale: mode === 'live' ? (parseFloat($('sel-speed').value) || 1.0) : 1.0,
    start_row: parseInt($('start-row').value || '0', 10),
    limit: mode === 'playback' && $('limit').value ? parseInt($('limit').value, 10) : null,
    source: mode,
    seed: parseInt($('seed').value || '42', 10),
    duration_s: $('duration').value ? parseFloat($('duration').value) : null,
    anomaly_every: $('anomevery').value ? parseInt($('anomevery').value, 10) : null,
  };
  const r = await fetch('/api/session/start', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await jsonResp(r);
  if (!r.ok) { showBanner(j.error || 'Не удалось запустить', false); return; }
  state.predCount = 0; state.actual = []; state.markers = []; state.preds = null;
  state.forecasts = [];   // история прогнозов {tick, vals[[horizon][n_targets]]}
  state.rows = [];        // сырые строки для плавающих показателей
  state.uncertain = { mean: null, std: null, alert: false };
  refreshLiveSelect();
  refreshOnLines();   // чипы доп. линий данных (сброс/фильтр по новому CSV)
  if (!state.u) makeUPlot();
  else rebuildOnline();   // пересоздание с текущим набором линий
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

// ------------------------------------------------- SNAPSHOT ТЕСТ/ОНЛАЙН
// PNG собирается в браузере: фиксируем ровно то, что видно — линии, цвета,
// текущее окно. Сохраняется в results/snapshots/ через сервер.
async function saveChartPng(u, wrapEl, name) {
  const canvases = [...wrapEl.querySelectorAll('canvas')].filter(c => c.width > 0);
  if (!canvases.length) { showBanner('График пуст — нечего сохранять', false); return; }
  const w = Math.max(10, wrapEl.clientWidth - 8), h = canvases[0].clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const off = document.createElement('canvas');
  off.width = Math.round(w * dpr);
  off.height = Math.round(h * dpr);
  const ctx = off.getContext('2d');
  const bg = (getComputedStyle(document.body).getPropertyValue('--bg') || '').trim() || '#0e131b';
  ctx.scale(dpr, dpr);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  for (const c of canvases) ctx.drawImage(c, 0, 0, w, h);
  const b64 = off.toDataURL('image/png').split(',')[1];
  const r = await fetch('/api/snapshots/save', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ name, png_b64: b64 }) });
  const j = await jsonResp(r);
  if (!r.ok) { showBanner(j.error || 'Ошибка снапшота', false); return; }
  showBanner(`✓ Snapshot сохранён: ${j.png}`);
}

$('btn-pr-snap')?.addEventListener('click', async () => {
  if (!state.prChart) { showBanner('Сначала выполните прогноз (🔮)', false); return; }
  const ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  await saveChartPng(state.prChart, $('pr-chart'),
    `predict_${($('pr-model').value || 'модель').slice(0, 40)}_${ts}`);
});

$('btn-on-snap')?.addEventListener('click', async () => {
  if (!state.u) { showBanner('Сначала постройте график сессии (▶ Старт)', false); return; }
  const ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  await saveChartPng(state.u, $('chart-wrap'), `online_${ts}`);
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
    // сохраняем полный прогноз (все цели × горизонт) для серии «прогноз был»
    state.forecasts.push({ tick: m.tick, vals: m.preds });
    if (state.forecasts.length > HISTORY + HORIZON * 2) state.forecasts.shift();
  }
  if (m.row) {
    state.rows.push({ ...m.row, __tick: m.tick });   // tick — для доп. линий графика
    if (state.rows.length > HISTORY + HORIZON) state.rows.shift();
  }
  drawOnline();
  drawLive();
  $('st-tick').textContent = m.tick;
  $('st-pred').textContent = state.predCount;
  $('st-ms').textContent = m.inference_ms ? m.inference_ms.toFixed(0) + ' мс' : '—';
  $('st-ratio').textContent = m.error_ratio == null ? '—' : m.error_ratio.toFixed(2);
  $('st-inp').className = 'marker ' + (m.input_anomaly ? 'on' : 'off');
  $('st-err').className = 'marker ' + (m.error_anomaly ? 'on' : 'off');
  // N4/C4: статус «не верить прогнозу» (вход СППР)
  if (m.uncertain_mean != null) state.uncertain.mean = m.uncertain_mean;
  if (m.uncertain_std != null) state.uncertain.std = m.uncertain_std;
  state.uncertain.alert = !!m.uncertain_alert;
  const unc = $('st-unc');
  if (unc) unc.className = 'marker ' + (m.uncertain_alert ? 'on' : 'off');
}

// ------------------------------------------------- ПЛАВАЮЩИЕ ПОКАЗАТЕЛИ (F1)
// Карточки-индикаторы выбранных параметров записи: live-значение на текущем
// tick, изменение за окно (60 tick), min/max за окно. Не только цель графика —
// любые колонки записи (вход для СППР-виджетов, CONCEPT §4.2).
const LIVE_WINDOW = 60;

async function refreshLiveSelect() {
  const csv = $('sel-csv').value;
  if (!csv) return;
  const r = await fetch(`/api/dataset/columns?csv=${encodeURIComponent(csv)}`);
  const j = await r.json();
  const sel = $('live-param');
  const keep = sel.value;
  sel.innerHTML = (j.columns || []).map(c => `<option>${c}</option>`).join('');
  if (keep && (j.columns || []).includes(keep)) sel.value = keep;
}

function addLiveCard() {
  const p = $('live-param').value;
  if (!p) return;
  if (state.live.includes(p)) { showBanner('Такая карточка уже есть', false); return; }
  state.live.push(p);
  renderLiveCards();
}

// ------------------------------------------------- ДОП. ЛИНИИ ОНЛАЙН-ГРАФИКА
// Чипы-чекбоксы: какие колонки записи рисовать на графике сессии вдобавок
// к цели (волна/скорость/руль + качка). Данные берутся из уже полученных
// строк tick-сообщений — пересчитывать ничего не нужно.
async function refreshOnLines() {
  const csv = $('sel-csv').value;
  const wrap = $('on-lines');
  if (!csv || !wrap) return;
  const r = await fetch(`/api/dataset/columns?csv=${encodeURIComponent(csv)}`);
  const j = await r.json();
  const cols = j.columns || [];
  state.onExtra = state.onExtra.filter(c => cols.includes(c));
  if (!state.onExtra.length) {
    // дефолт: высота волны + скорость судна (пример заказчика)
    const def = ['Wave.Highest(метры)', 'SOG(узлы)'].filter(c => cols.includes(c));
    state.onExtra = def;
  }
  wrap.innerHTML = '';
  for (const c of cols) {
    const lab = document.createElement('label');
    lab.className = state.onExtra.includes(c) ? 'on' : '';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.onExtra.includes(c);
    cb.onchange = () => {
      if (cb.checked) { if (!state.onExtra.includes(c)) state.onExtra.push(c); }
      else state.onExtra = state.onExtra.filter(x => x !== c);
      lab.className = state.onExtra.includes(c) ? 'on' : '';
      rebuildOnline();
    };
    const txt = document.createElement('span');
    txt.textContent = c;
    lab.appendChild(cb);
    lab.appendChild(txt);
    wrap.appendChild(lab);
  }
}

function removeLiveCard(p) {
  state.live = state.live.filter(x => x !== p);
  renderLiveCards();
}

function renderLiveCards() {
  const wrap = $('live-cards');
  wrap.innerHTML = state.live.map(p => `
    <div class="live-card" data-param="${p}">
      <div class="live-name">${p}<button class="live-x" title="убрать">×</button></div>
      <div class="live-val">—</div>
      <div class="live-stat">Δ60: — · min: — · max: —</div>
    </div>`).join('') || '<span style="color:var(--dim);font-size:12px">карточек нет — выберите параметры выше</span>';
  wrap.querySelectorAll('.live-x').forEach(b =>
    b.addEventListener('click', () => removeLiveCard(b.closest('.live-card').dataset.param)));
}

function drawLive() {
  if (!state.live.length) return;
  const cur = state.rows[state.rows.length - 1];
  if (!cur) return;
  const win = state.rows.slice(-LIVE_WINDOW);
  for (const p of state.live) {
    const card = document.querySelector(`.live-card[data-param="${p}"]`);
    if (!card) continue;
    const v = cur[p];
    const vals = win.map(r => r[p]).filter(x => x != null);
    card.querySelector('.live-val').textContent =
      v == null ? '—' : (+v).toFixed(3);
    if (vals.length > 1) {
      const delta = (+vals[vals.length - 1]) - (+vals[0]);
      const mn = Math.min(...vals), mx = Math.max(...vals);
      card.querySelector('.live-stat').textContent =
        `Δ${LIVE_WINDOW}: ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} · min: ${mn.toFixed(3)} · max: ${mx.toFixed(3)}`;
    }
  }
}

$('btn-live-add')?.addEventListener('click', addLiveCard);

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

// ------------------------------------------------------------------ ДАННЫЕ
let vwChart = null, vwData = null;

function vwRangeSec() {
  const v = $('vw-range').value;
  if (v === 'custom') return parseInt($('vw-custom').value || '0', 10);
  return parseInt(v, 10);
}

async function showView() {
  const csv = $('vw-csv').value;
  const start = parseInt($('vw-start').value || '0', 10);
  const dur = vwRangeSec();
  const r = await fetch(`/api/dataset/view?csv=${encodeURIComponent(csv)}&start=${start}&duration=${dur}`);
  const j = await r.json();
  if (j.error) { showBanner(j.error, false); return; }
  vwData = j;
  $('vw-total').textContent = j.total_rows;
  // чипы выбора параметров: несколько линий на одном графике
  if ($('vw-targets').dataset.for !== csv) {
    $('vw-targets').dataset.for = csv;
    // выбор сохраняем, если колонки нового дата-сета его содержат
    state.vwSel = state.vwSel.filter(c => j.columns.includes(c));
    if (!state.vwSel.length) {
      // дефолт: качка + высота волны + скорость судна (запрос заказчика)
      const def = ['Roll(градусы)', 'Vertical(Метр)', 'SOG(узлы)']
        .filter(c => j.columns.includes(c));
      state.vwSel = def.length ? def : [j.columns[0]];
    }
  }
  renderTargetChips(j.columns);
  drawView();
}

function renderTargetChips(columns) {
  const wrap = $('vw-targets');
  wrap.innerHTML = '';
  for (const c of columns) {
    const lab = document.createElement('label');
    lab.className = state.vwSel.includes(c) ? 'on' : '';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.vwSel.includes(c);
    cb.onchange = () => {
      if (cb.checked) { if (!state.vwSel.includes(c)) state.vwSel.push(c); }
      else state.vwSel = state.vwSel.filter(x => x !== c);
      lab.className = state.vwSel.includes(c) ? 'on' : '';
      drawView();
    };
    const txt = document.createElement('span');
    txt.textContent = c;
    lab.appendChild(cb);
    lab.appendChild(txt);
    wrap.appendChild(lab);
  }
}

function vwColor(col) {
  if (!state.vwColors[col]) {
    const used = Object.keys(state.vwColors).length;
    state.vwColors[col] = VW_PALETTE[used % VW_PALETTE.length];
  }
  return state.vwColors[col];
}

function drawView() {
  if (!vwData || !state.vwSel.length) { $('vw-chips').innerHTML = ''; return; }
  if ($('vw-chart').clientWidth < 20) return;   // вкладка скрыта — график создадим при открытии
  // фикс подписи: график пересоздаётся с актуальными названиями и цветами
  // (uPlot не обновляет легенду при смене series.label на лету —
  // поэтому подпись оставалась «roll» при переключении параметра)
  if (vwChart) {
    for (const ser of vwChart.series) {
      if (!ser.label || ser.label === 'с') continue;
      if (ser.show === false) state.vwHidden[ser.label] = true;
      else delete state.vwHidden[ser.label];
    }
    vwChart.destroy();
    vwChart = null;
  }
  const series = [{ label: 'с' }];
  const data = [vwData.xs];
  for (const col of state.vwSel) {
    series.push({ label: col, stroke: vwColor(col), width: 1.5,
                  points: { show: vwData.n_points <= 200, size: 2 },
                  show: !state.vwHidden[col] });
    data.push(vwData.series[col] || []);
  }
  const c = makeCursorOpts();
  vwChart = new uPlot({
    width: Math.max(10, $('vw-chart').clientWidth - 8), height: 280,
    scales: { x: { time: false } },
    axes: axesFromTheme(),
    cursor: c.cursor, hooks: c.hooks,
    series,
  }, data, $('vw-chart'));
  vwChart.__readout = $('vw-readout');
  attachZoomReset(vwChart, $('btn-vw-reset'));
  bindChips(vwChart, 'vw-chips', state.vwColors);
  window.vwChart = vwChart; // отладка: доступ из консоли
}

$('btn-vw-show')?.addEventListener('click', showView);
$('vw-csv')?.addEventListener('change', () => {
  $('vw-targets').dataset.for = ''; showView();
});
$('vw-range')?.addEventListener('change', () => {
  $('vw-custom-wrap').style.display = $('vw-range').value === 'custom' ? 'flex' : 'none';
  showView();
});
$('vw-custom')?.addEventListener('change', showView);

$('btn-vw-snap')?.addEventListener('click', async () => {
  if (!vwData || !state.vwSel.length) { showBanner('Сначала постройте график (👁 Показать)', false); return; }
  const body = {
    csv: $('vw-csv').value,
    start: parseInt($('vw-start').value || '0', 10),
    duration: vwRangeSec(),
    targets: state.vwSel,
    colors: state.vwColors,
    dark: !isLight(),
  };
  const r = await fetch('/api/dataset/snapshot', { method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const j = await jsonResp(r);
  if (!r.ok) { showBanner(j.error || 'Ошибка снапшота', false); return; }
  $('vw-snapresult').innerHTML =
    `✓ Сохранено: <code>${j.png}</code> и <code>${j.csv}</code> (${j.rows} строк)`;
});

$('btn-upload')?.addEventListener('click', async () => {
  const f = $('dt-file').files[0];
  if (!f) { showBanner('Выберите файл', false); return; }
  const text = await f.text();
  const r = await fetch('/api/datasets/upload', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ name: f.name, data_b64: btoa(unescape(encodeURIComponent(text))) }) });
  const j = await jsonResp(r);
  if (!r.ok) { showBanner(j.error || 'Ошибка загрузки', false); return; }
  showBanner(`Загружено: ${j.file} (${j.rows} строк, ${j.columns} колонок)`);
  refreshData(); loadCsvs(['tr-csv', 'sel-csv', 'pr-csv']);
});

$('btn-syn')?.addEventListener('click', async () => {
  const r = await fetch('/api/synthetic', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ rows: parseInt($('sy-rows').value || '30000', 10),
                           seed: parseInt($('sy-seed').value || '42', 10) }) });
  const j = await jsonResp(r);
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
  const j = await jsonResp(r);
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
  initTheme();
  initHelp();
  const cfg = await jsonGet('/api/config');
  const tsel = $('tr-profile');
  cfg.profiles.forEach(p => {
    const o = document.createElement('option'); o.value = o.textContent = p;
    tsel.appendChild(o);
  });
  tsel.value = cfg.profile;
  await Promise.all([loadModels(['sel-model', 'pr-model', 'ft-base', 'tr-init']),
                     loadCsvs(['tr-csv', 'sel-csv', 'pr-csv', 'vw-csv']),
                     refreshOverview(), refreshData(), refreshRegistry(), refreshJobs()]);
  // просмотр данных: график строится при каждом открытии вкладки
  // (мультивыбор/цвета пересоздают график, лениво при скрытой вкладке)
  document.querySelector('.tab[data-page="data"]').addEventListener('click', () => {
    showView();
  });
  showView();
  // график онлайна создаётся лениво — при первом открытии вкладки «Онлайн»
  const st = await jsonGet('/api/session/status');
  if (st.running) { document.querySelector('.tab[data-page="online"]').click(); makeUPlot(); connectWs(); }
  setInterval(() => { refreshJobs(); }, 5000);
}
init();
