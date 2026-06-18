'use strict';

/* ═══ CONFIG ═══════════════════════════════════════════════════════════════ */
const QUARTER_SECS = 10 * 60;
const QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4', 'OT'];
const STORAGE_KEY = 'titans_exp1_v1';

const DEFAULT_PLAYERS = [
  'Aaron Breziner', 'Andre Setton', 'Zury Attia', 'Joseph Gabay',
  'Saul Piciotto', 'Daniel Abadi', 'Ilay Mendelson', 'Alberto Yahni',
  'Ramon Malca', 'Ariel Gean', 'Ariel Ghershfeld', 'Toby Burstein',
];

const SHOT_CFG = [
  { label: '2 Puntos',   madeKey: '2PT_MADE', attKey: '2PT_ATT', pts: 2, madeColor: '#1a5e35', missColor: '#5c0a0a' },
  { label: '3 Puntos',   madeKey: '3PT_MADE', attKey: '3PT_ATT', pts: 3, madeColor: '#0e4d3f', missColor: '#5c0a0a' },
  { label: 'Tiro Libre', madeKey: 'FT_MADE',  attKey: 'FT_ATT',  pts: 1, madeColor: '#0b5e4e', missColor: '#5c0a0a' },
];

const STAT_CFG = [
  { key: 'REB_OFF', label: 'Reb. Ofensivo',  color: '#8a6914' },
  { key: 'REB_DEF', label: 'Reb. Defensivo', color: '#7d5d12' },
  { key: 'AST',     label: 'Asistencia',     color: '#1f6090' },
  { key: 'TOV',     label: 'Pérdida',        color: '#7b241c' },
  { key: 'BLK',     label: 'Bloqueo',        color: '#0e6251' },
  { key: 'FOUL',    label: 'Falta',          color: '#a93226' },
];

const ALL_STAT_KEYS = ['2PT_MADE', '2PT_ATT', '3PT_MADE', '3PT_ATT', 'FT_MADE', 'FT_ATT',
  'REB_OFF', 'REB_DEF', 'AST', 'TOV', 'BLK', 'FOUL'];

/* STAT key → readable label for event feed */
const STAT_LABELS = {
  '2PT_MADE': '✓ 2PT', '2PT_MISS': '✗ 2PT',
  '3PT_MADE': '✓ 3PT', '3PT_MISS': '✗ 3PT',
  'FT_MADE':  '✓ TL',  'FT_MISS':  '✗ TL',
  'REB_OFF': 'Reb. Ofensivo', 'REB_DEF': 'Reb. Defensivo',
  'AST': 'Asistencia ⚠', 'TOV': 'Pérdida',
  'BLK': 'Bloqueo', 'FOUL': 'Falta',
};

/* ═══ STATE ════════════════════════════════════════════════════════════════ */
let S = newState();
let sessionId = crypto.randomUUID();
let ws = null;
let aiEvents = [];
let eventCount = 0;
let jerseyMap = {};          // {"7": "Joseph Gabay"} — learned/set by user
let smartScanActive = false;

function newState() {
  const players = [...DEFAULT_PLAYERS];
  return {
    gameName: 'Titans vs ___',
    quarter: 'Q1',
    secsLeft: QUARTER_SECS,
    clockRunning: false,
    players,
    stats: makeStats(players),
    minutesPlayed: makeMinutes(players),
    onCourt: [],
    fouledOut: {},
    selected: players[0],
    history: [],
    rivalScore: 0,
    rivalFouls: 0,
  };
}

function makeStats(players) {
  const obj = {};
  players.forEach(p => {
    obj[p] = {};
    ALL_STAT_KEYS.forEach(k => obj[p][k] = 0);
  });
  return obj;
}

function makeMinutes(players) {
  const obj = {};
  players.forEach(p => obj[p] = 0);
  return obj;
}

function ensurePlayer(p) {
  if (!S.stats[p]) S.stats[p] = {};
  ALL_STAT_KEYS.forEach(k => { if (S.stats[p][k] === undefined) S.stats[p][k] = 0; });
  if (S.minutesPlayed[p] === undefined) S.minutesPlayed[p] = 0;
}

/* ═══ HELPERS ══════════════════════════════════════════════════════════════ */
function pts(p) {
  const s = S.stats[p];
  return (s['2PT_MADE'] || 0) * 2 + (s['3PT_MADE'] || 0) * 3 + (s['FT_MADE'] || 0);
}
function totStat(k) { return S.players.reduce((n, p) => n + (S.stats[p]?.[k] || 0), 0); }
function totalPts() { return S.players.reduce((n, p) => n + pts(p), 0); }
function totalMins() { return Object.values(S.minutesPlayed).reduce((a, b) => a + b, 0); }
function fgPct(p) {
  const s = S.stats[p];
  const a = (s['2PT_ATT'] || 0) + (s['3PT_ATT'] || 0);
  return a > 0 ? ((s['2PT_MADE'] || 0) + (s['3PT_MADE'] || 0)) / a : null;
}
function ftPct(p) {
  const s = S.stats[p];
  return s['FT_ATT'] > 0 ? s['FT_MADE'] / s['FT_ATT'] : null;
}
function threePct(p) {
  const s = S.stats[p];
  return s['3PT_ATT'] > 0 ? s['3PT_MADE'] / s['3PT_ATT'] : null;
}
function fmtPct(v) { return v === null ? '--' : Math.round(v * 100) + '%'; }
function fmtMin(secs) {
  const m = Math.floor(secs / 60), s = secs % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}
function shortName(full) {
  const parts = full.trim().split(/\s+/);
  if (parts.length === 1) return full;
  const first = parts[0];
  const hasDup = S.players.some(p => p !== full && p.split(/\s+/)[0] === first);
  return hasDup ? `${first} ${parts[1].slice(0, 2)}.` : first;
}

/* ═══ CLOCK ════════════════════════════════════════════════════════════════ */
let _interval = null;

function startInterval() {
  if (_interval) return;
  _interval = setInterval(tick, 1000);
}

function tick() {
  if (!S.clockRunning) return;
  S.onCourt.forEach(p => S.minutesPlayed[p] = (S.minutesPlayed[p] || 0) + 1);
  if (S.secsLeft > 0) {
    S.secsLeft--;
    if (S.secsLeft === 0) { S.clockRunning = false; onQuarterEnd(); }
  }
  renderClock();
  refreshMinCells();
}

function onQuarterEnd() {
  updateClockBtn();
  toast(`¡Fin del ${S.quarter}! 🏀`);
}

function toggleClock() {
  S.clockRunning = !S.clockRunning;
  updateClockBtn();
  save();
}

function resetClock() {
  S.clockRunning = false;
  S.secsLeft = QUARTER_SECS;
  updateClockBtn();
  renderClock();
}

function updateClockBtn() {
  const btn = document.getElementById('btnClock');
  if (!btn) return;
  if (S.clockRunning) { btn.textContent = '⏸'; btn.classList.add('paused'); }
  else { btn.textContent = '▶'; btn.classList.remove('paused'); }
}

function renderClock() {
  const el = document.getElementById('clockDisplay');
  if (!el) return;
  const m = Math.floor(S.secsLeft / 60), s = S.secsLeft % 60;
  el.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  el.className = 'clock-display';
  if (S.secsLeft <= 60) el.classList.add('red');
  else if (S.secsLeft <= 180) el.classList.add('yellow');
}

/* ═══ RENDER ═══════════════════════════════════════════════════════════════ */
function renderAll() {
  document.getElementById('gameName').value = S.gameName;
  renderQuarterBtns();
  document.getElementById('quarterDisplay').textContent = S.quarter;
  renderClock();
  updateClockBtn();
  renderPlayers();
  renderStatButtons();
  renderActiveBadge();
  updateScores();
  renderTable();
}

function updateScores() {
  const el = document.getElementById('titansScore');
  if (el) el.textContent = totalPts();
  const rv = document.getElementById('rivalScore');
  if (rv) rv.textContent = S.rivalScore || 0;
  const rf = document.getElementById('rivalFouls');
  if (rf) rf.textContent = S.rivalFouls || 0;
}

function renderQuarterBtns() {
  const sel = document.getElementById('quarterBtns');
  if (!sel) return;
  sel.innerHTML = '';
  QUARTERS.forEach(q => {
    const b = document.createElement('button');
    b.textContent = q; b.dataset.q = q;
    if (S.quarter === q) b.classList.add('active');
    b.addEventListener('click', () => {
      S.quarter = q;
      document.getElementById('quarterDisplay').textContent = q;
      document.querySelectorAll('.quarter-btns button').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.q === q)
      );
      save();
    });
    sel.appendChild(b);
  });
}

function renderPlayers() {
  const list = document.getElementById('playerList');
  if (!list) return;
  list.innerHTML = '';
  S.players.forEach(p => {
    const wrap = document.createElement('div');
    wrap.className = 'player-wrap';

    const btn = document.createElement('button');
    btn.className = 'player-btn';
    btn.dataset.player = p;
    btn.textContent = shortName(p);
    if (p === S.selected) btn.classList.add('selected');
    if (S.fouledOut[p]) btn.classList.add('fouled-out');
    else if (S.onCourt.includes(p)) btn.classList.add('on-court');

    const cBtn = document.createElement('button');
    cBtn.dataset.court = p;
    if (S.fouledOut[p]) {
      cBtn.className = 'court-btn fouled-out';
      cBtn.textContent = '✕ 5F';
    } else if (S.onCourt.includes(p)) {
      cBtn.className = 'court-btn in';
      cBtn.textContent = '● Cancha';
    } else {
      cBtn.className = 'court-btn';
      cBtn.textContent = '○ Fuera';
    }

    // Jersey number input (for Smart Scan auto-detection)
    const jerseyNum = Object.entries(jerseyMap).find(([n, name]) => name === p)?.[0] || '';
    const jerseyInput = document.createElement('input');
    jerseyInput.className = 'jersey-input';
    jerseyInput.type = 'number';
    jerseyInput.min = '0';
    jerseyInput.max = '99';
    jerseyInput.placeholder = '#';
    jerseyInput.title = 'Jersey number (helps AI identify who scored)';
    jerseyInput.value = jerseyNum;
    jerseyInput.addEventListener('change', () => {
      const num = jerseyInput.value.trim();
      // Remove old mapping for this player
      Object.keys(jerseyMap).forEach(k => { if (jerseyMap[k] === p) delete jerseyMap[k]; });
      if (num) jerseyMap[num] = p;
      saveState();
    });

    wrap.append(btn, jerseyInput, cBtn);
    list.appendChild(wrap);
  });
}

function renderStatButtons() {
  const grid = document.getElementById('statGrid');
  if (!grid) return;
  grid.innerHTML = '';

  const shotLbl = document.createElement('div');
  shotLbl.className = 'stat-section-label';
  shotLbl.textContent = 'TIROS';
  grid.appendChild(shotLbl);

  SHOT_CFG.forEach(cfg => {
    const s = S.selected ? S.stats[S.selected] : null;
    const madeVal = s ? (s[cfg.madeKey] || 0) : 0;
    const missVal = s ? ((s[cfg.attKey] || 0) - (s[cfg.madeKey] || 0)) : 0;

    const makeBtn = document.createElement('button');
    makeBtn.className = 'stat-btn';
    makeBtn.dataset.stat = cfg.madeKey;
    makeBtn.style.setProperty('--btn-color', cfg.madeColor);
    makeBtn.innerHTML = `<span class="stat-label">✓ ${cfg.label}</span><span class="stat-count" id="sc_${cfg.madeKey}">${madeVal}</span>`;

    const missBtn = document.createElement('button');
    missBtn.className = 'stat-btn';
    missBtn.dataset.stat = cfg.attKey + '_MISS';
    missBtn.style.setProperty('--btn-color', cfg.missColor);
    missBtn.innerHTML = `<span class="stat-label">✗ ${cfg.label}</span><span class="stat-count" id="sc_${cfg.attKey}_MISS">${missVal}</span>`;

    grid.append(makeBtn, missBtn);
  });

  const otherLbl = document.createElement('div');
  otherLbl.className = 'stat-section-label';
  otherLbl.textContent = 'ESTADÍSTICAS';
  grid.appendChild(otherLbl);

  STAT_CFG.forEach(cfg => {
    const btn = document.createElement('button');
    btn.className = 'stat-btn';
    btn.dataset.stat = cfg.key;
    btn.style.setProperty('--btn-color', cfg.color);
    const count = S.selected ? (S.stats[S.selected]?.[cfg.key] ?? 0) : 0;
    btn.innerHTML = `<span class="stat-label">${cfg.label}</span><span class="stat-count" id="sc_${cfg.key}">${count}</span>`;
    if (cfg.key === 'FOUL') btn.style.gridColumn = '1 / -1';
    grid.appendChild(btn);
  });
}

function updateCounts() {
  if (!S.selected) return;
  const s = S.stats[S.selected];
  SHOT_CFG.forEach(cfg => {
    const mEl = document.getElementById(`sc_${cfg.madeKey}`);
    const xEl = document.getElementById(`sc_${cfg.attKey}_MISS`);
    if (mEl) mEl.textContent = s[cfg.madeKey] || 0;
    if (xEl) xEl.textContent = (s[cfg.attKey] || 0) - (s[cfg.madeKey] || 0);
  });
  STAT_CFG.forEach(cfg => {
    const el = document.getElementById(`sc_${cfg.key}`);
    if (el) el.textContent = s[cfg.key] || 0;
  });
}

function renderActiveBadge() {
  const nameEl = document.getElementById('activeName');
  const statusEl = document.getElementById('activeStatus');
  if (nameEl) nameEl.textContent = S.selected || '—';
  if (statusEl) {
    if (S.fouledOut[S.selected]) {
      statusEl.textContent = '✕ Eliminado'; statusEl.className = 'status-dot fouled-out';
    } else if (S.onCourt.includes(S.selected)) {
      statusEl.textContent = '● En cancha'; statusEl.className = 'status-dot in';
    } else {
      statusEl.textContent = '○ Fuera'; statusEl.className = 'status-dot out';
    }
  }
}

/* ═══ STAT ACTIONS ══════════════════════════════════════════════════════════ */
function logStat(key) {
  if (!S.selected) { toast('Selecciona un jugador'); return; }
  if (S.fouledOut[S.selected]) { toast(`${shortName(S.selected)} eliminado`); return; }

  S.history.push({
    player: S.selected, key,
    snap: JSON.stringify(S.stats),
    fouledOutSnap: JSON.stringify(S.fouledOut),
    onCourtSnap: JSON.stringify(S.onCourt),
  });
  if (S.history.length > 60) S.history.shift();

  applyStatKey(S.selected, key);
  if (key === 'FOUL') checkFoulOut(S.selected);

  updateCounts();
  updateScores();
  updateTableRow(S.selected);
  animateBtn(key);
  save();
}

function applyStatKey(player, key) {
  const st = S.stats[player];
  if (key === '2PT_MADE')     { st['2PT_MADE']++; st['2PT_ATT']++; }
  else if (key === '2PT_ATT_MISS') { st['2PT_ATT']++; }
  else if (key === '3PT_MADE')     { st['3PT_MADE']++; st['3PT_ATT']++; }
  else if (key === '3PT_ATT_MISS') { st['3PT_ATT']++; }
  else if (key === 'FT_MADE')      { st['FT_MADE']++; st['FT_ATT']++; }
  else if (key === 'FT_ATT_MISS')  { st['FT_ATT']++; }
  else if (key === '2PT_MISS')     { st['2PT_ATT']++; }
  else if (key === '3PT_MISS')     { st['3PT_ATT']++; }
  else if (key === 'FT_MISS')      { st['FT_ATT']++; }
  else { st[key]++; }
}

function animateBtn(key) {
  const btn = document.querySelector(`.stat-btn[data-stat="${CSS.escape(key)}"]`);
  if (!btn) return;
  btn.classList.remove('tapped');
  void btn.offsetWidth;
  btn.classList.add('tapped');
  btn.addEventListener('animationend', () => btn.classList.remove('tapped'), { once: true });
}

function checkFoulOut(player) {
  if ((S.stats[player].FOUL || 0) < 5) return;
  S.fouledOut[player] = true;
  const idx = S.onCourt.indexOf(player);
  if (idx !== -1) S.onCourt.splice(idx, 1);
  renderPlayers();
  if (player === S.selected) renderActiveBadge();
  toast(`⚠️ ${shortName(player)} eliminado por 5 faltas`);
}

function undoLast() {
  if (!S.history.length) { toast('Nada que deshacer'); return; }
  const { player, snap, fouledOutSnap, onCourtSnap } = S.history.pop();
  S.stats = JSON.parse(snap);
  S.fouledOut = JSON.parse(fouledOutSnap);
  S.onCourt = JSON.parse(onCourtSnap);
  if (player === S.selected) updateCounts();
  updateScores();
  updateTableRow(player);
  renderPlayers();
  if (player === S.selected) renderActiveBadge();
  save();
}

function selectPlayer(name) {
  S.selected = name;
  document.querySelectorAll('.player-btn').forEach(b =>
    b.classList.toggle('selected', b.dataset.player === name)
  );
  renderActiveBadge();
  updateCounts();
  document.querySelectorAll('#statsTable tbody tr:not(.total-row)').forEach((row, i) => {
    row.classList.toggle('selected', S.players[i] === name);
  });
}

function toggleCourt(player) {
  if (S.fouledOut[player]) { toast(`${shortName(player)} fue eliminado`); return; }
  const idx = S.onCourt.indexOf(player);
  if (idx === -1) S.onCourt.push(player);
  else S.onCourt.splice(idx, 1);
  renderPlayers();
  if (player === S.selected) renderActiveBadge();
  save();
}

function adjustRival(n) {
  S.rivalScore = Math.max(0, (S.rivalScore || 0) + n);
  updateScores();
  save();
}

function adjustRivalFoul(n) {
  S.rivalFouls = Math.max(0, (S.rivalFouls || 0) + n);
  updateScores();
  save();
}

/* ═══ STATS TABLE ═══════════════════════════════════════════════════════════ */
const TABLE_COLS = [
  { h: 'MIN', fn: p => fmtMin(S.minutesPlayed[p] || 0), total: () => fmtMin(totalMins()) },
  { h: 'PTS', fn: p => pts(p), total: () => S.players.reduce((n, p) => n + pts(p), 0) },
  { h: '2PT M/A', fn: p => `${S.stats[p]['2PT_MADE']}/${S.stats[p]['2PT_ATT']}`, total: () => `${totStat('2PT_MADE')}/${totStat('2PT_ATT')}` },
  { h: '3PT M/A', fn: p => `${S.stats[p]['3PT_MADE']}/${S.stats[p]['3PT_ATT']}`, total: () => `${totStat('3PT_MADE')}/${totStat('3PT_ATT')}` },
  { h: 'TL M/A',  fn: p => `${S.stats[p]['FT_MADE']}/${S.stats[p]['FT_ATT']}`,  total: () => `${totStat('FT_MADE')}/${totStat('FT_ATT')}` },
  { h: 'FG%',  fn: p => fmtPct(fgPct(p)), total: () => { const m = totStat('2PT_MADE') + totStat('3PT_MADE'), a = totStat('2PT_ATT') + totStat('3PT_ATT'); return fmtPct(a > 0 ? m / a : null); } },
  { h: 'FT%',  fn: p => fmtPct(ftPct(p)), total: () => { const m = totStat('FT_MADE'), a = totStat('FT_ATT'); return fmtPct(a > 0 ? m / a : null); } },
  { h: 'R.Of', fn: p => S.stats[p].REB_OFF, total: () => totStat('REB_OFF') },
  { h: 'R.Def', fn: p => S.stats[p].REB_DEF, total: () => totStat('REB_DEF') },
  { h: 'REB',  fn: p => S.stats[p].REB_OFF + S.stats[p].REB_DEF, total: () => totStat('REB_OFF') + totStat('REB_DEF') },
  { h: 'AST*', fn: p => S.stats[p].AST, total: () => totStat('AST'), disclaimer: true },
  { h: 'TOV',  fn: p => S.stats[p].TOV, total: () => totStat('TOV') },
  { h: 'BLQ',  fn: p => S.stats[p].BLK, total: () => totStat('BLK') },
  { h: 'FALT', fn: p => S.stats[p].FOUL, total: () => totStat('FOUL'),
    cellClass: p => S.stats[p].FOUL >= 5 ? 'falt-5' : S.stats[p].FOUL >= 4 ? 'falt-4' : '' },
];

function renderTable() {
  const tbl = document.getElementById('statsTable');
  if (!tbl) return;
  const headers = ['Jugador', ...TABLE_COLS.map(c => c.h)];
  let html = '<thead><tr>' + headers.map(h => `<th title="${h === 'AST*' ? 'Asistencias — precisión ~65% con AI. Se recomienda confirmar manualmente.' : h}">${h}</th>`).join('') + '</tr></thead>';
  if (TABLE_COLS.some(c => c.disclaimer)) {
    html += '<caption style="caption-side:bottom;font-size:0.65rem;color:#aaa;padding:4px 0">* AST: precisión AI ~65% — verifica con Quick Stats</caption>';
  }
  html += '<tbody>';
  S.players.forEach(p => {
    const cls = [p === S.selected ? 'selected' : '', S.fouledOut[p] ? 'row-fouled-out' : ''].filter(Boolean).join(' ');
    html += `<tr${cls ? ` class="${cls}"` : ''}><td>${shortName(p)}${S.fouledOut[p] ? '*' : ''}</td>`;
    TABLE_COLS.forEach(col => {
      const val = col.fn(p);
      const cc = col.cellClass ? col.cellClass(p) : '';
      html += `<td${cc ? ` class="${cc}"` : ''}>${val}</td>`;
    });
    html += '</tr>';
  });
  html += '<tr class="total-row"><td>TOTAL</td>' + TABLE_COLS.map(col => `<td>${col.total()}</td>`).join('') + '</tr></tbody>';
  tbl.innerHTML = html;
}

function updateTableRow(player) {
  const sec = document.getElementById('tableSection');
  if (!sec || sec.classList.contains('hidden')) return;
  renderTable();
}

function refreshMinCells() {
  const sec = document.getElementById('tableSection');
  if (!sec || sec.classList.contains('hidden')) return;
  const rows = document.querySelectorAll('#statsTable tbody tr:not(.total-row)');
  rows.forEach((row, i) => {
    const p = S.players[i];
    if (p) row.cells[1].textContent = fmtMin(S.minutesPlayed[p] || 0);
  });
}

/* ═══ AI EVENT FEED ══════════════════════════════════════════════════════════ */
function addAiEvent(data) {
  aiEvents.push(data);
  eventCount++;
  document.getElementById('feedCounter').textContent = `${eventCount} event${eventCount !== 1 ? 's' : ''} detected`;

  const feed = document.getElementById('eventFeed');
  const empty = feed.querySelector('.feed-empty');
  if (empty) empty.remove();

  const card = buildEventCard(data);
  feed.insertBefore(card, feed.firstChild);

  // Auto-scroll to top
  feed.scrollTop = 0;

  // Auto-confirm if high confidence (≥ 0.88)
  if ((data.confidence || 0) >= 0.88) {
    setTimeout(() => confirmEvent(data.id, card), 800);
  }
}

function buildEventCard(data) {
  const card = document.createElement('div');
  card.className = 'event-card pending';
  card.id = `ev_${data.id}`;

  const isRival = data.team === 'rival' || data.player === 'RIVAL';
  const playerName = isRival ? 'RIVAL' : data.player;
  const statLabel = STAT_LABELS[data.stat] || data.stat;
  const conf = data.confidence || 0;
  const confClass = conf >= 0.8 ? 'high' : conf >= 0.6 ? 'med' : 'low';
  const confPct = Math.round(conf * 100);

  // Source badge styling
  const sourceBadge = {
    'whistle_vision': '🚨 Whistle',
    'audio':          '🎙 Audio',
    'full_auto':      '🤖 Vision',
  }[data.source] || '🤖 AI';

  const isFoul = data.stat === 'FOUL';

  card.innerHTML = `
    <span class="event-source ${data.source === 'whistle_vision' ? 'whistle' : 'ai'}">${sourceBadge}</span>
    <span class="event-ts">${data.video_ts || ''}</span>
    <div class="event-body">
      <span class="event-player ${isRival ? 'event-rival' : ''} ${isFoul ? 'event-foul' : ''}">${playerName}</span>
      <span class="event-stat"> — ${statLabel}</span>
      ${data.quote ? `<div class="event-quote">"${data.quote}"</div>` : ''}
    </div>
    <span class="event-conf ${confClass}">${confPct}%</span>
    <div class="event-actions">
      <button class="btn-confirm" onclick="confirmEvent('${data.id}', this.closest('.event-card'))">✓</button>
      <button class="btn-reject"  onclick="rejectEvent('${data.id}', this.closest('.event-card'))">✗</button>
    </div>`;
  return card;
}

function confirmEvent(id, card) {
  const ev = aiEvents.find(e => e.id === id);
  if (!ev || ev._confirmed) return;
  ev._confirmed = true;

  const isRival = ev.team === 'rival' || ev.player === 'RIVAL';

  if (isRival) {
    // Apply rival stat
    if (ev.stat === '2PT_MADE') adjustRival(2);
    else if (ev.stat === '3PT_MADE') adjustRival(3);
    else if (ev.stat === 'FT_MADE') adjustRival(1);
    else if (ev.stat === 'FOUL') adjustRivalFoul(1);
  } else {
    // Apply Titans player stat
    const player = ev.player;
    if (S.players.includes(player)) {
      const prev = S.selected;
      S.selected = player;
      applyStatKey(player, ev.stat);
      if (ev.stat === 'FOUL') checkFoulOut(player);
      S.selected = prev;
      updateCounts();
      updateScores();
      updateTableRow(player);
      save();
    }
  }

  if (card) {
    card.className = 'event-card confirmed';
    card.querySelector('.event-actions').innerHTML = '<span style="color:#2ecc71;font-size:0.78rem">✓ Added</span>';
  }
  toast(`✓ ${ev.player} — ${STAT_LABELS[ev.stat] || ev.stat}`);
}

function rejectEvent(id, card) {
  const ev = aiEvents.find(e => e.id === id);
  if (ev) ev._confirmed = false;
  if (card) {
    card.className = 'event-card rejected';
    card.querySelector('.event-actions').innerHTML = '<span style="color:#e74c3c;font-size:0.78rem">✗ Skipped</span>';
  }
}

function confirmAllHighConfidence() {
  const pending = aiEvents.filter(e => !e._confirmed && (e.confidence || 0) >= 0.75);
  pending.forEach(ev => {
    const card = document.getElementById(`ev_${ev.id}`);
    confirmEvent(ev.id, card);
  });
  toast(`✓ Confirmed ${pending.length} events`);
}

/* ═══ WEBSOCKET ══════════════════════════════════════════════════════════════ */
function connectWS() {
  if (ws) { ws.close(); ws = null; }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/${sessionId}`);

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleServerMsg(msg);
  };

  ws.onclose = () => { ws = null; };
  ws.onerror = () => toast('WebSocket error');
}

function handleServerMsg(msg) {
  switch (msg.type) {
    case 'status':
      setStatus(msg.msg, msg.level || 'info');
      break;
    case 'progress':
      setProgress(msg.pct);
      if (msg.done) setTimeout(() => hideStatus(), 3000);
      break;
    case 'vision_progress':
      setProgress(msg.pct);
      setStatus(`📷 Scanning frames... ${msg.frame}/${msg.total} (${msg.pct}%)`, 'info');
      break;
    case 'scan_tick':
      setProgress(msg.pct);
      if (msg.changed) setStatus(`⚡ Score change @${msg.video_ts} — analyzing play...`, 'info');
      break;
    case 'jersey_update': {
      // Merge AI-learned jersey numbers (named players) into our map
      const unknownNums = [];
      Object.entries(msg.map || {}).forEach(([num, val]) => {
        if (num.startsWith('_')) return; // skip meta keys like _titans_color
        if (S.players.includes(val)) {
          if (!jerseyMap[num]) {
            jerseyMap[num] = val;
            toast(`🔍 Aprendido: #${num} = ${val}`);
          }
        } else if (val && (val.includes('TITANS') || val.includes('Titans') || val.includes('gray')) && !jerseyMap[num]) {
          // AI saw a Titans jersey number but doesn't know the player — ask user to assign
          unknownNums.push(num);
        }
      });
      // Show "unassigned Titans numbers" banner if we have some
      if (unknownNums.length > 0) {
        showUnassignedJerseys(unknownNums);
      }
      renderPlayers();
      break;
    }
    case 'smart_scan_started':
      smartScanActive = true;
      document.getElementById('btnSmartScan').classList.add('hidden');
      document.getElementById('btnStopSmartScan').classList.remove('hidden');
      break;
    case 'smart_scan_done':
      smartScanActive = false;
      document.getElementById('btnSmartScan').classList.remove('hidden');
      document.getElementById('btnStopSmartScan').classList.add('hidden');
      if (msg.learned_jerseys) {
        Object.assign(jerseyMap, msg.learned_jerseys);
        renderPlayers();
      }
      break;
    case 'auto_progress': {
      setProgress(msg.pct);
      const phaseLabel = {download: '⬇ Downloading', scanning: '🔍 Scanning', audio: '🎙 Audio'}[msg.phase] || '';
      if (phaseLabel) setStatus(`${phaseLabel}... ${msg.pct}%`, 'info');
      break;
    }
    case 'auto_phase':
      break;
    case 'auto_done': {
      const btn = document.getElementById('btnFullAuto');
      if (btn) { btn.disabled = false; btn.textContent = '🚀 Full Auto'; }
      document.getElementById('btnStopAuto').classList.add('hidden');
      if (msg.learned_jerseys) {
        Object.assign(jerseyMap, msg.learned_jerseys);
        renderPlayers();
      }
      break;
    }
    case 'video_info':
      if (msg.title) document.getElementById('gameName').value = `Titans vs ${msg.title.slice(0, 30)}`;
      break;
    case 'ai_event':
      addAiEvent(msg);
      break;
    case 'score_update':
      applyScoreUpdate(msg);
      break;
    case 'substitution': {
      const inP  = msg.sub_in  || '?';
      const outP = msg.sub_out || '?';
      setStatus(`🔄 Sustitución @${msg.video_ts}: ${outP} sale → ${inP} entra`, 'info');
      // Update on-court status in state
      if (msg.sub_out && S.players.includes(msg.sub_out)) {
        const idx = S.onCourt.indexOf(msg.sub_out);
        if (idx !== -1) S.onCourt.splice(idx, 1);
      }
      if (msg.sub_in && S.players.includes(msg.sub_in) && !S.onCourt.includes(msg.sub_in)) {
        S.onCourt.push(msg.sub_in);
      }
      renderPlayers();
      saveState();
      break;
    }
    case 'minutes_update': {
      const mins = msg.minutes || {};
      Object.entries(mins).forEach(([player, m]) => {
        if (!S.stats[player]) return;
        if (m > (S.stats[player].MIN || 0)) {
          S.stats[player].MIN = m;
        }
      });
      updateTableRow && S.players.forEach(p => updateTableRow(p));
      break;
    }
    case 'whistle_events':
      setStatus(`🎵 Audio: ${msg.count} whistles + ${msg.cheer_count} crowd cheers ready for analysis`, 'info');
      break;
    case 'timeout_stats': {
      // Broadcast stat overlay found during timeout/halftime
      const pstats = msg.player_stats || {};
      const label = msg.is_halftime ? '🏁 Halftime stats' : '⏸ Timeout stats';
      setStatus(`${label} @${msg.video_ts} — ${Object.keys(pstats).length} player(s) detected`, 'success');
      // Auto-apply accumulated cumulative stats visible in the overlay
      let applied = 0;
      Object.entries(pstats).forEach(([playerName, stats]) => {
        const matched = S.players.find(p => p.toLowerCase().includes(playerName.toLowerCase()) ||
                                           playerName.toLowerCase().includes(p.split(' ')[0].toLowerCase()));
        if (!matched) return;
        if (!S.stats[matched]) S.stats[matched] = {};
        const statMap = { PTS: null, REB: 'REB_DEF', AST: 'AST', BLK: 'BLK', FOUL: 'FOUL' };
        Object.entries(stats || {}).forEach(([k, v]) => {
          if (v != null && statMap[k]) {
            const sk = statMap[k];
            if (sk && (S.stats[matched][sk] || 0) < v) {
              S.stats[matched][sk] = v;
              applied++;
            }
          }
        });
      });
      if (applied > 0) {
        saveState();
        renderStats();
        toast(`📊 ${label}: ${applied} stat(s) synced from broadcast overlay!`);
      }
      break;
    }
    case 'error':
      setStatus(`❌ ${msg.msg}`, 'error');
      break;
    case 'heartbeat':
      break;
  }
}

function applyScoreUpdate(msg) {
  if (msg.titans_score !== null && msg.titans_score !== undefined) {
    const el = document.getElementById('titansScore');
    // Only update if AI score is higher than current (monotonic — scores only go up)
    if (el && msg.titans_score > parseInt(el.textContent || '0')) {
      toast(`📷 Score updated: Titans ${msg.titans_score} | ${msg.quarter || ''} ${msg.clock || ''}`);
    }
  }
  if (msg.rival_score !== null && msg.rival_score !== undefined) {
    S.rivalScore = Math.max(S.rivalScore || 0, msg.rival_score);
    updateScores();
  }
}

function setStatus(msg, level = 'info') {
  const bar = document.getElementById('statusBar');
  const msgEl = document.getElementById('statusMsg');
  if (bar) bar.classList.remove('hidden');
  if (msgEl) {
    msgEl.textContent = msg;
    msgEl.style.color = level === 'error' ? '#e74c3c' : level === 'success' ? '#2ecc71' : level === 'warn' ? '#f1c40f' : '#a0a0c0';
  }
}

function setProgress(pct) {
  const fill = document.getElementById('progressFill');
  if (fill) fill.style.width = pct + '%';
}

function hideStatus() {
  document.getElementById('statusBar')?.classList.add('hidden');
}

/* ═══ YOUTUBE ═════════════════════════════════════════════════════════════════ */
function extractVideoId(url) {
  const m = url.match(/(?:v=|youtu\.be\/|embed\/)([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
}

function loadVideo(url) {
  const vid = extractVideoId(url);
  if (!vid) { toast('Invalid YouTube URL'); return false; }
  const frame = document.getElementById('ytFrame');
  const placeholder = document.getElementById('videoPlaceholder');
  frame.src = `https://www.youtube.com/embed/${vid}?autoplay=0`;
  frame.classList.remove('hidden');
  if (placeholder) placeholder.classList.add('hidden');
  return true;
}

async function startTracking() {
  const url = document.getElementById('ytUrl').value.trim();
  if (!url) { toast('Enter a YouTube URL'); return; }

  if (!loadVideo(url)) return;

  document.getElementById('btnStart').classList.add('hidden');
  document.getElementById('btnStop').classList.remove('hidden');
  document.getElementById('statusBar').classList.remove('hidden');
  setStatus('Connecting...', 'info');
  setProgress(0);

  // Clear previous AI events
  aiEvents = [];
  eventCount = 0;
  document.getElementById('feedCounter').textContent = '0 events detected';
  document.getElementById('eventFeed').innerHTML = '<div class="feed-empty">Analyzing game...</div>';

  connectWS();

  // Give WS a moment to connect
  await new Promise(r => setTimeout(r, 500));

  const resp = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, session_id: sessionId, players: S.players }),
  });
  const data = await resp.json();
  if (data.error) {
    setStatus(`❌ ${data.error}`, 'error');
    document.getElementById('btnStart').classList.remove('hidden');
    document.getElementById('btnStop').classList.add('hidden');
  }
}

async function stopTracking() {
  await fetch('/api/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  document.getElementById('btnStart').classList.remove('hidden');
  document.getElementById('btnStop').classList.add('hidden');
  setStatus('Stopped.', 'warn');
}

/* ═══ REPORT ══════════════════════════════════════════════════════════════════ */
function pct1(v) { return v === null ? '--' : (Math.round(v * 1000) / 10) + '%'; }

function generateReport() {
  const rivalPtsRaw = prompt('¿Cuántos puntos anotó el rival? (vacío si ya está en el marcador)', String(S.rivalScore || ''));
  const rivalPts = rivalPtsRaw !== null && rivalPtsRaw.trim() !== '' ? parseInt(rivalPtsRaw) : (S.rivalScore || null);
  const teamPts = totalPts();
  const gn = S.gameName;
  const dateStr = new Date().toLocaleDateString('es-ES', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const timeStr = new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });

  const sorted = [...S.players].sort((a, b) => pts(b) - pts(a));
  const topThree = sorted.filter(p => pts(p) > 0).slice(0, 3);

  const playerRowsHtml = S.players.map(p => {
    const s = S.stats[p];
    const min = S.minutesPlayed[p] || 0;
    const noPlay = min === 0 && pts(p) === 0;
    let nota = '';
    if (noPlay) nota = 'No jugó';
    else if (S.fouledOut[p]) nota = '⚠️ Eliminado (5F)';
    else if ((s.FOUL || 0) >= 4) nota = `⚠️ ${s.FOUL} faltas`;
    else if (p === topThree[0] && pts(p) > 0) nota = '⭐ MVP';
    const rowStyle = S.fouledOut[p] ? ' style="color:#c0392b"' : noPlay ? ' style="color:#888"' : '';
    return `<tr${rowStyle}>
      <td>${p}${S.fouledOut[p] ? '*' : ''}</td>
      <td>${fmtMin(min)}</td><td><strong>${pts(p)}</strong></td>
      <td>${s['2PT_MADE'] || 0}/${s['2PT_ATT'] || 0}</td>
      <td>${s['3PT_MADE'] || 0}/${s['3PT_ATT'] || 0}</td>
      <td>${s['FT_MADE'] || 0}/${s['FT_ATT'] || 0}</td>
      <td>${pct1(fgPct(p))}</td><td>${pct1(ftPct(p))}</td>
      <td>${s.TOV || 0}</td><td>${(s.REB_OFF || 0) + (s.REB_DEF || 0)}</td>
      <td>${s.AST || 0}</td><td>${s.FOUL || 0}</td>
      <td style="font-size:0.8em;color:#555">${nota}</td>
    </tr>`;
  }).join('');

  const topPlayersHtml = topThree.map((p, i) => {
    const s = S.stats[p];
    const label = i === 0 ? '⭐ MVP del partido' : `${i + 1}° Anotador`;
    const bullets = [
      `${pts(p)} puntos anotados`, `${fmtMin(S.minutesPlayed[p] || 0)} en cancha`,
      fgPct(p) !== null ? `${pct1(fgPct(p))} en tiros de campo` : null,
      threePct(p) !== null && (s['3PT_ATT'] || 0) > 0 ? `${pct1(threePct(p))} en triples` : null,
      ftPct(p) !== null && (s['FT_ATT'] || 0) > 0 ? `${pct1(ftPct(p))} en tiros libres` : null,
      `${s.TOV || 0} pérdidas`, `${s.FOUL || 0} falta${(s.FOUL || 0) !== 1 ? 's' : ''}`,
    ].filter(Boolean);
    return `<div class="top-player"><div class="top-player-name">${i + 1}. ${p} — <em>${label}</em></div><ul>${bullets.map(b => `<li>${b}</li>`).join('')}</ul></div>`;
  }).join('') || '<p>Sin datos.</p>';

  const ftA = totStat('FT_ATT'), ftM = totStat('FT_MADE');
  const thA = totStat('3PT_ATT'), thM = totStat('3PT_MADE');
  const teamFgM = totStat('2PT_MADE') + totStat('3PT_MADE');
  const teamFgA = totStat('2PT_ATT') + totStat('3PT_ATT');

  const recs = [];
  if (ftA >= 5 && ftM / ftA < 0.50) recs.push(`Práctica de tiros libres: equipo al ${pct1(ftM / ftA)} — meta mínima 60%.`);
  if (thA === 0 || (thA > 0 && thM / thA < 0.25)) recs.push('Desarrollar el juego perimetral: identificar tiradores de 3 puntos.');
  if (totStat('TOV') > 15) recs.push(`Reducir pérdidas de balón: ${totStat('TOV')} turnovers.`);
  const nearFO = S.players.filter(p => (S.stats[p].FOUL || 0) >= 4 && !S.fouledOut[p]);
  if (nearFO.length) recs.push(`Control de faltas: ${nearFO.map(shortName).join(', ')} terminó con 4+ faltas.`);
  if (!recs.length) recs.push('Buen partido. Mantener el nivel de ejecución.');

  const scoreBlock = rivalPts !== null
    ? `<div class="score-block"><div class="score-team"><div class="score-name">TITANS</div><div class="score-pts">${teamPts}</div></div><div class="score-vs">VS</div><div class="score-team"><div class="score-name">RIVAL</div><div class="score-pts">${rivalPts}</div></div></div>`
    : `<div class="score-block"><div class="score-team"><div class="score-name">TITANS</div><div class="score-pts">${teamPts}</div></div></div>`;

  let summaryTxt = '';
  if (rivalPts !== null) {
    const diff = teamPts - rivalPts;
    summaryTxt += diff > 0 ? `Los Titans se impusieron por ${teamPts} a ${rivalPts}, logrando una victoria por ${diff} puntos. ` : diff < 0 ? `Los Titans cayeron por ${rivalPts} a ${teamPts}. ` : `Empate ${teamPts}-${rivalPts}. `;
  }
  if (topThree[0]) summaryTxt += `El equipo se apoyó en ${topThree[0]} (${pts(topThree[0])} PT${fgPct(topThree[0]) !== null ? `, ${pct1(fgPct(topThree[0]))} FG` : ''}). `;
  if (topThree[1]) summaryTxt += `${topThree[1]} fue el segundo anotador con ${pts(topThree[1])} puntos. `;
  if (ftA >= 5 && ftM / ftA < 0.50) summaryTxt += `El equipo mostró debilidades en el tiro libre (${pct1(ftM / ftA)}), un área clave a trabajar. `;

  const html = `<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Reporte — ${gn}</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:'Helvetica Neue',Arial,sans-serif;background:#f5f5f5;color:#1a1a2e;font-size:14px}.page{max-width:820px;margin:0 auto;background:#fff;padding:40px}.report-header{text-align:center;border-bottom:3px solid #1a1a2e;padding-bottom:20px;margin-bottom:24px}.report-header .emoji{font-size:2.5rem}.report-header h1{font-size:1.1rem;letter-spacing:.15em;color:#555;margin:4px 0}.report-header h2{font-size:1.8rem;font-weight:900;color:#1a1a2e;margin:6px 0}.report-header .meta{font-size:.8rem;color:#888;margin-top:6px}.score-block{display:flex;justify-content:center;align-items:center;gap:30px;margin:16px 0}.score-team{text-align:center}.score-name{font-size:.9rem;font-weight:700;letter-spacing:.1em;color:#555}.score-pts{font-size:3.5rem;font-weight:900;color:#1a1a2e;line-height:1}.score-vs{font-size:1.2rem;font-weight:700;color:#888}section{margin-bottom:28px}section h3{font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#fff;background:#1a1a2e;padding:6px 12px;border-radius:4px;margin-bottom:12px}section p{line-height:1.65;color:#333}.team-table{width:100%;border-collapse:collapse}.team-table td{padding:8px 12px;border-bottom:1px solid #eee}.team-table td:first-child{color:#555;width:55%}.team-table td:last-child{font-weight:700;font-size:1.05rem}.stats-table{width:100%;border-collapse:collapse;font-size:.82rem}.stats-table th{background:#1a1a2e;color:#f1c40f;font-weight:700;padding:6px 8px;text-align:center;white-space:nowrap}.stats-table th:first-child{text-align:left}.stats-table td{padding:5px 8px;text-align:center;border-bottom:1px solid #eee}.stats-table td:first-child{text-align:left;font-weight:600}.stats-table tr:nth-child(even) td{background:#f9f9f9}.stats-table .total-row td{background:#f0f0f0;font-weight:700;border-top:2px solid #1a1a2e}.top-player{margin-bottom:14px}.top-player-name{font-weight:700;font-size:1rem;margin-bottom:4px}.top-player ul{padding-left:20px}.top-player li{line-height:1.7;color:#444}.rec-list{padding-left:20px}.rec-list li{line-height:1.8;color:#333}.print-btn{display:block;margin:0 auto 32px;padding:12px 32px;background:#1a1a2e;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:700;cursor:pointer}@media print{.print-btn{display:none}}</style>
</head><body><div class="page">
<button class="print-btn" onclick="window.print()">🖨️ Guardar como PDF / Imprimir</button>
<div class="report-header"><div class="emoji">🏀</div><h1>REPORTE DE PARTIDO</h1><h2>${gn}</h2>${scoreBlock}<div class="meta">${dateStr} · ${timeStr}</div></div>
<section><h3>Resumen Ejecutivo</h3><p>${summaryTxt || 'Partido completado.'}</p></section>
<section><h3>Estadísticas Generales del Equipo</h3><table class="team-table">
<tr><td>Puntos totales</td><td>${teamPts}</td></tr>
<tr><td>FG% (tiros de campo)</td><td>${teamFgA > 0 ? pct1(teamFgM / teamFgA) : '--'}</td></tr>
<tr><td>3PT% (triples)</td><td>${thA > 0 ? pct1(thM / thA) : '--'} (${thM}/${thA})</td></tr>
<tr><td>FT% (tiros libres)</td><td>${ftA > 0 ? pct1(ftM / ftA) : '--'} (${ftM}/${ftA}) ${ftA >= 5 && ftM / ftA < 0.5 ? '⚠️' : ''}</td></tr>
<tr><td>Rebotes totales</td><td>${totStat('REB_OFF') + totStat('REB_DEF')} (Of: ${totStat('REB_OFF')} / Def: ${totStat('REB_DEF')})</td></tr>
<tr><td>Asistencias totales *</td><td>${totStat('AST')}</td></tr>
<tr><td>Pérdidas de balón</td><td>${totStat('TOV')}</td></tr>
<tr><td>Bloqueos</td><td>${totStat('BLK')}</td></tr>
<tr><td>Faltas totales</td><td>${totStat('FOUL')}</td></tr>
${rivalPts !== null ? `<tr><td>Diferencia final</td><td>${teamPts - rivalPts > 0 ? '+' : ''}${teamPts - rivalPts} ${teamPts > rivalPts ? '✅' : '❌'}</td></tr>` : ''}
</table></section>
<section><h3>Estadísticas Individuales</h3><table class="stats-table"><thead><tr><th>Jugador</th><th>MIN</th><th>PT</th><th>2PT M/A</th><th>3PT M/A</th><th>TL M/A</th><th>FG%</th><th>FT%</th><th>TO</th><th>REB</th><th title="Asistencias — precisión AI ~65%. Se recomienda verificar manualmente.">AST *</th><th>FALT</th><th>Notas</th></tr></thead><tbody>${playerRowsHtml}</tbody>
<tfoot><tr class="total-row"><td>TOTAL</td><td>${fmtMin(totalMins())}</td><td>${teamPts}</td><td>${totStat('2PT_MADE')}/${totStat('2PT_ATT')}</td><td>${totStat('3PT_MADE')}/${totStat('3PT_ATT')}</td><td>${totStat('FT_MADE')}/${totStat('FT_ATT')}</td><td>${teamFgA > 0 ? pct1(teamFgM / teamFgA) : '--'}</td><td>${ftA > 0 ? pct1(ftM / ftA) : '--'}</td><td>${totStat('TOV')}</td><td>${totStat('REB_OFF') + totStat('REB_DEF')}</td><td>${totStat('AST')}</td><td>${totStat('FOUL')}</td><td></td></tr></tfoot></table>
<p style="font-size:0.7rem;color:#777;margin-top:4px">* AST: asistencias detectadas por visión AI — precisión aproximada 65%. Se recomienda confirmar con registro manual.</p></section>
<section><h3>Jugadores Destacados</h3>${topPlayersHtml}</section>
<section><h3>Recomendaciones para el Próximo Partido</h3><ul class="rec-list">${recs.map(r => `<li>${r}</li>`).join('')}</ul></section>
</div></body></html>`;

  const w = window.open('', '_blank');
  if (w) { w.document.write(html); w.document.close(); }
  else toast('Permite ventanas emergentes');
}

/* ═══ PERSIST ═════════════════════════════════════════════════════════════════ */
let _saveTimer = null;
function save() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    const { history, ...toSave } = S;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
  }, 1500);
}

function loadSaved() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const d = JSON.parse(raw);
    S = { ...newState(), ...d, clockRunning: false, history: [] };
    S.players.forEach(ensurePlayer);
    DEFAULT_PLAYERS.forEach(p => { if (!S.players.includes(p)) { S.players.push(p); ensurePlayer(p); } });
    return true;
  } catch { return false; }
}

/* ═══ PLAYER MANAGEMENT ═════════════════════════════════════════════════════ */
function addPlayer() {
  const raw = prompt('Nombre completo del jugador:');
  if (!raw) return;
  const name = raw.trim().replace(/\w\S*/g, w => w[0].toUpperCase() + w.slice(1).toLowerCase());
  if (S.players.includes(name)) { toast(`${name} ya existe`); return; }
  S.players.push(name);
  ensurePlayer(name);
  renderPlayers();
  renderTable();
  save();
}

function removePlayer() {
  if (!S.selected || S.players.length <= 1) return;
  if (!confirm(`¿Quitar a ${S.selected}?`)) return;
  const idx = S.players.indexOf(S.selected);
  S.players.splice(idx, 1);
  delete S.stats[S.selected];
  delete S.minutesPlayed[S.selected];
  S.onCourt = S.onCourt.filter(p => p !== S.selected);
  S.selected = S.players[Math.max(0, idx - 1)];
  renderAll();
  save();
}

function newGame() {
  if (!confirm('¿Nuevo partido? Se borrará el partido actual.')) return;
  const opp = prompt('Nombre del rival:', '___') || '___';
  S = newState();
  S.gameName = `Titans vs ${opp.trim()}`;
  aiEvents = [];
  eventCount = 0;
  document.getElementById('feedCounter').textContent = '0 events detected';
  document.getElementById('eventFeed').innerHTML = '<div class="feed-empty">No events yet.</div>';
  localStorage.removeItem(STORAGE_KEY);
  sessionId = crypto.randomUUID();
  renderAll();
  toast('🔄 Nuevo partido iniciado');
}

/* ═══ TOAST ═══════════════════════════════════════════════════════════════════ */
let _toastTimer = null;
function toast(msg) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

let _shownUnassigned = new Set();
function showUnassignedJerseys(nums) {
  const newNums = nums.filter(n => !_shownUnassigned.has(n) && !jerseyMap[n]);
  if (newNums.length === 0) return;
  newNums.forEach(n => _shownUnassigned.add(n));

  const banner = document.createElement('div');
  banner.className = 'unassigned-banner';
  banner.innerHTML = `<strong>🔍 AI vio camiseta(s) Titans sin asignar: ${newNums.map(n=>`<b>#${n}</b>`).join(', ')}</strong>
    <span style="font-size:0.82em;opacity:0.8"> — asigna en el panel de jugadores (campo #)</span>
    <button onclick="this.parentElement.remove()" style="float:right;background:none;border:none;cursor:pointer;font-size:1.1em">✕</button>`;
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 15000);
}

/* ═══ INIT ════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  loadSaved();
  renderAll();
  startInterval();

  // Bind events
  document.getElementById('gameName').addEventListener('change', e => { S.gameName = e.target.value; save(); });
  document.getElementById('btnClock').addEventListener('click', toggleClock);
  document.getElementById('btnResetClock').addEventListener('click', resetClock);
  document.getElementById('btnStart').addEventListener('click', startTracking);
  document.getElementById('btnStop').addEventListener('click', stopTracking);

  // ── Full Auto — the one button to rule them all ──────────────────────────
  document.getElementById('btnFullAuto').addEventListener('click', async () => {
    const url = document.getElementById('ytUrl').value.trim();
    if (!url) { toast('Paste a YouTube URL first'); return; }

    // Warn if jersey numbers are missing (they're critical for player ID)
    const jerseyCount = Object.keys(jerseyMap).filter(k => !k.startsWith('_')).length;
    if (jerseyCount === 0 && S.players.length > 0) {
      const proceed = confirm(
        '⚠️ Sin números de camiseta registrados.\n\n' +
        'Para identificar jugadores automáticamente, ingresa el # de camiseta de cada jugador en el panel (campo # al lado del nombre).\n\n' +
        '¿Continuar de todas formas? (La IA usará perfiles físicos, pero será menos preciso)'
      );
      if (!proceed) return;
    }

    connectWS();
    await new Promise(r => setTimeout(r, 400));

    const btn = document.getElementById('btnFullAuto');
    btn.disabled = true;
    btn.textContent = '⏳ Starting...';
    document.getElementById('btnStopAuto').classList.remove('hidden');

    // Read AI setup panel values
    const titansColor = (document.getElementById('titansColor')?.value || 'gray/white').trim();
    const rivalColor  = (document.getElementById('rivalColor')?.value  || 'colored').trim();
    // Parse player hints textarea into {"player name": "description"} dict
    const playerProfiles = {};
    const hintsRaw = document.getElementById('playerHints')?.value || '';
    hintsRaw.split('\n').forEach(line => {
      const idx = line.indexOf(':');
      if (idx > 0) {
        const pname = line.slice(0, idx).trim();
        const hint  = line.slice(idx + 1).trim();
        if (pname && hint) playerProfiles[pname] = hint;
      }
    });

    const r = await fetch('/api/full-auto', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        session_id: sessionId,
        players: S.players,
        jersey_map: jerseyMap,
        score_interval: 3,
        player_profiles: playerProfiles,
        titans_jersey_color: titansColor,
        rival_jersey_color: rivalColor,
      }),
    });
    const d = await r.json();
    if (d.error) {
      toast(`❌ ${d.error}`);
      btn.disabled = false;
      btn.textContent = '🚀 Full Auto';
      document.getElementById('btnStopAuto').classList.add('hidden');
      return;
    }
    btn.textContent = '🔄 Running...';
    setStatus('🚀 Full Auto started — warmup scan → jersey detection → full game analysis...', 'info');
  });

  document.getElementById('btnStopAuto').addEventListener('click', async () => {
    await fetch('/api/stop-full-auto', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const btn = document.getElementById('btnFullAuto');
    btn.disabled = false;
    btn.textContent = '🚀 Full Auto';
    document.getElementById('btnStopAuto').classList.add('hidden');
    setStatus('Full Auto stopped.', 'warn');
  });
  document.getElementById('btnReport').addEventListener('click', generateReport);
  document.getElementById('btnNewGame').addEventListener('click', newGame);
  document.getElementById('btnUndo').addEventListener('click', undoLast);
  document.getElementById('btnAdd').addEventListener('click', addPlayer);
  document.getElementById('btnRemove').addEventListener('click', removePlayer);
  document.getElementById('btnConfirmAll').addEventListener('click', confirmAllHighConfidence);

  // ── Quick Stats Panel ─────────────────────────────────────────────────────
  let qsSelectedPlayer = null;

  function renderQsPlayers() {
    const container = document.getElementById('qsPlayers');
    if (!container) return;
    container.innerHTML = '';
    S.players.forEach(p => {
      const btn = document.createElement('button');
      btn.className = 'qs-player-btn' + (p === qsSelectedPlayer ? ' selected' : '') + (S.onCourt.includes(p) ? ' on-court' : '');
      btn.textContent = shortName(p);
      btn.addEventListener('click', () => {
        qsSelectedPlayer = (qsSelectedPlayer === p) ? null : p;
        document.getElementById('qsSelected').textContent =
          qsSelectedPlayer ? `✓ ${qsSelectedPlayer} — tap a stat` : '← Select a player first';
        renderQsPlayers();
      });
      container.appendChild(btn);
    });
  }

  function logQuickStat(player, stat) {
    if (!player || !S.players.includes(player)) { toast('Select a player first'); return; }
    // Push to history for undo
    S.history.push({ player, key: stat, prev: { ...S.stats[player] } });
    applyStatKey(player, stat);
    if (stat === 'FOUL') checkFoulOut(player);
    updateCounts();
    updateScores();
    updateTableRow(player);
    save();

    // Add to event feed as a manual event
    const id = crypto.randomUUID().slice(0, 8);
    const feedEv = {
      id, source: 'manual', video_ts: currentVideoTs(),
      player, team: 'titans', stat, confidence: 1.0,
      quote: 'Manual quick stat', _confirmed: true,
    };
    aiEvents.push(feedEv);
    eventCount++;
    document.getElementById('feedCounter').textContent = `${eventCount} event${eventCount !== 1 ? 's' : ''} detected`;
    const feed = document.getElementById('eventFeed');
    const empty = feed.querySelector('.feed-empty');
    if (empty) empty.remove();
    const card = buildEventCard(feedEv);
    card.className = 'event-card confirmed';
    card.querySelector('.event-actions').innerHTML = '<span style="color:#2ecc71;font-size:0.78rem">✓ Manual</span>';
    feed.insertBefore(card, feed.firstChild);

    toast(`✓ ${shortName(player)} — ${STAT_LABELS[stat] || stat}`);

    // Visual feedback on the stat button
    const statBtn = document.querySelector(`.qs-stat-btn[data-stat="${CSS.escape(stat)}"]`);
    if (statBtn) {
      statBtn.classList.add('tapped');
      statBtn.addEventListener('animationend', () => statBtn.classList.remove('tapped'), { once: true });
    }
  }

  function currentVideoTs() {
    // Try to read current clock from scoreboard display
    const clock = document.getElementById('clockDisplay')?.textContent || '';
    const quarter = document.getElementById('quarterDisplay')?.textContent || '';
    return quarter && clock ? `${quarter} ${clock}` : new Date().toLocaleTimeString();
  }

  document.querySelectorAll('.qs-stat-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!qsSelectedPlayer) { toast('Select a player first ↑'); return; }
      logQuickStat(qsSelectedPlayer, btn.dataset.stat);
    });
  });

  document.getElementById('btnToggleQS')?.addEventListener('click', () => {
    const body = document.querySelector('.qs-body');
    const btn = document.getElementById('btnToggleQS');
    if (body.classList.toggle('hidden')) { btn.textContent = 'Show'; }
    else { btn.textContent = 'Hide'; }
  });

  // Re-render QS players whenever roster changes
  const _origRenderPlayers = renderPlayers;
  renderPlayers = function() { _origRenderPlayers(); renderQsPlayers(); };
  renderQsPlayers();

  document.getElementById('btnToggleTable').addEventListener('click', () => {
    const sec = document.getElementById('tableSection');
    sec.classList.toggle('hidden');
    if (!sec.classList.contains('hidden')) renderTable();
  });
  document.getElementById('btnCloseTable').addEventListener('click', () =>
    document.getElementById('tableSection').classList.add('hidden')
  );

  // Vision controls
  document.getElementById('btnAnalyzeFrame').addEventListener('click', async () => {
    const url = document.getElementById('ytUrl').value.trim();
    if (!url) { toast('Enter a YouTube URL first'); return; }
    const tsRaw = document.getElementById('frameTs').value.trim();
    let timestamp = 60;
    if (tsRaw) {
      const parts = tsRaw.split(':').map(Number);
      timestamp = parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0];
    }
    connectWS();
    await new Promise(r => setTimeout(r, 300));
    const r = await fetch('/api/analyze-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, session_id: sessionId, players: S.players, timestamp }),
    });
    const d = await r.json();
    if (d.error) toast(`❌ ${d.error}`);
    else setStatus(`📷 Analyzing frame @${Math.floor(timestamp/60)}:${String(timestamp%60).padStart(2,'0')}...`, 'info');
  });

  document.getElementById('btnStartScan').addEventListener('click', async () => {
    const url = document.getElementById('ytUrl').value.trim();
    if (!url) { toast('Enter a YouTube URL first'); return; }
    if (!confirm('Auto Scan downloads frames every 45 seconds for the full game.\nThis may take a while. Continue?')) return;
    connectWS();
    await new Promise(r => setTimeout(r, 300));
    const r = await fetch('/api/start-vision-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, session_id: sessionId, players: S.players, interval: 45 }),
    });
    const d = await r.json();
    if (d.error) { toast(`❌ ${d.error}`); return; }
    document.getElementById('btnStartScan').classList.add('hidden');
    document.getElementById('btnStopScan').classList.remove('hidden');
    setStatus('📷 Auto scan started...', 'info');
  });

  document.getElementById('btnStopScan').addEventListener('click', async () => {
    await fetch('/api/stop-vision-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    document.getElementById('btnStartScan').classList.remove('hidden');
    document.getElementById('btnStopScan').classList.add('hidden');
    setStatus('Scan stopped.', 'warn');
  });

  // Smart Scan — fully automatic play-by-play detection
  document.getElementById('btnSmartScan').addEventListener('click', async () => {
    const url = document.getElementById('ytUrl').value.trim();
    if (!url) { toast('Enter a YouTube URL first'); return; }

    const jerseyCount = Object.keys(jerseyMap).length;
    const msg = jerseyCount > 0
      ? `Smart Scan will auto-detect all plays using Claude Vision.\n${jerseyCount} jersey number(s) configured.\n\nStart?`
      : `Smart Scan will auto-detect plays using Claude Vision.\n\nTip: enter jersey numbers (#) next to each player for better accuracy.\n\nStart anyway?`;
    if (!confirm(msg)) return;

    connectWS();
    await new Promise(r => setTimeout(r, 400));

    const r = await fetch('/api/start-smart-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        session_id: sessionId,
        players: S.players,
        jersey_map: jerseyMap,
        poll_interval: 12,
        start_ts: 60,
      }),
    });
    const d = await r.json();
    if (d.error) { toast(`❌ ${d.error}`); return; }
    smartScanActive = true;
    document.getElementById('btnSmartScan').classList.add('hidden');
    document.getElementById('btnStopSmartScan').classList.remove('hidden');
    setStatus('🤖 Smart Scan running — watching every 12s for score changes...', 'info');
  });

  document.getElementById('btnStopSmartScan').addEventListener('click', async () => {
    await fetch('/api/stop-smart-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    smartScanActive = false;
    document.getElementById('btnSmartScan').classList.remove('hidden');
    document.getElementById('btnStopSmartScan').classList.add('hidden');
    setStatus('Smart Scan stopped.', 'warn');
  });

  document.getElementById('playerList').addEventListener('click', e => {
    const pb = e.target.closest('.player-btn');
    if (pb) { selectPlayer(pb.dataset.player); return; }
    const cb = e.target.closest('.court-btn');
    if (cb) toggleCourt(cb.dataset.court);
  });

  document.getElementById('statGrid').addEventListener('click', e => {
    const b = e.target.closest('.stat-btn');
    if (b) logStat(b.dataset.stat);
  });
});
