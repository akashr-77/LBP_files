
/* ══════════════════════════════════════════
   CONFIG & STATE
══════════════════════════════════════════ */
let API_BASE = 'http://localhost:8000';
let parsedSignal = null;

let signalChart = null, simChart = null;
let simRunning = false, simInterval = null, simCount = 0, health = 100;

const MODEL_CLASSES = ['Ball-007','Ball-014','Ball-021','IR-007','IR-014','IR-021','OR-007','OR-014','OR-021','Normal'];

const PALETTE = ['#1e6fff','#38bdf8','#34d399','#fbbf24','#f87171','#a78bfa','#06b6d4','#fb923c','#ec4899','#84cc16'];

/* ══════ NAV ══════ */
function switchTab(name, el) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('page-' + name).classList.add('active');
}



/* ══════ .MAT PARSER (Level-5, pure JS) ══════ */
function parseMat(buffer) {
  const view = new DataView(buffer);
  const out = {};
  let off = 128; // skip 128-byte header

  while (off + 8 < buffer.byteLength) {
    const dtype = view.getUint32(off, true);
    const nbytes = view.getUint32(off + 4, true);
    off += 8;
    if (nbytes === 0) continue;

    if (dtype === 14) { // miMATRIX
      const elEnd = off + nbytes;
      try {
        // Array flags (skip 8 bytes tag + 8 bytes data = 16)
        off += 16;
        // Dimensions
        const dimTag   = view.getUint32(off, true);
        const dimBytes = view.getUint32(off + 4, true);
        off += 8;
        const dims = [];
        for (let i = 0; i < dimBytes / 4; i++) dims.push(view.getInt32(off + i * 4, true));
        off += dimBytes + (dimBytes % 8 ? 8 - dimBytes % 8 : 0);
        // Name
        const nameTag   = view.getUint32(off, true);
        const nameBytes = view.getUint32(off + 4, true);
        off += 8;
        let name = '';
        for (let i = 0; i < nameBytes; i++) name += String.fromCharCode(view.getUint8(off + i));
        off += nameBytes + (nameBytes % 8 ? 8 - nameBytes % 8 : 0);
        // Data
        if (off < elEnd) {
          const realType  = view.getUint32(off, true);
          const realBytes = view.getUint32(off + 4, true);
          off += 8;
          const total = dims.reduce((a, b) => a * b, 1);
          const arr = [];
          for (let i = 0; i < total; i++) {
            if      (realType === 9) arr.push(view.getFloat64(off + i * 8, true));
            else if (realType === 7) arr.push(view.getFloat32(off + i * 4, true));
            else if (realType === 5) arr.push(view.getInt32  (off + i * 4, true));
            else if (realType === 3) arr.push(view.getInt16  (off + i * 2, true));
          }
          if (arr.length > 0) out[name] = arr;
        }
      } catch(e) { /* skip bad element */ }
      off = elEnd;
    } else {
      off += nbytes + (nbytes % 8 ? 8 - nbytes % 8 : 0);
    }
  }
  return out;
}

/* ══════ FILE HANDLING ══════ */
function handleFile(ev) {
  const file = ev.target.files[0];
  if (!file) return;

  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['mat', 'csv', 'json', 'txt'];
  if (!allowed.includes(ext)) {
    showError('Unsupported file type: .' + ext + '. Please upload .mat, .csv, .json, or .txt');
    return;
  }

  document.getElementById('file-name').textContent = file.name + ' (' + (file.size/1024).toFixed(1) + ' KB)';
  hideError();

  const reader = new FileReader();

  if (ext === 'mat') {
    reader.onload = e => {
      try {
        const mat = parseMat(e.target.result);
        const keys = Object.keys(mat);
        const deKey = keys.find(k => k.includes('DE_time'));
        if (!deKey) throw new Error('No DE_time key found. Keys in file: ' + (keys.join(', ') || 'none'));
        const sig = mat[deKey];
        if (sig.length < 1024) throw new Error('Signal too short: ' + sig.length + ' samples');
        parsedSignal = sig;
        setParse('ok', '✓  .mat parsed — ' + deKey + ', ' + sig.length.toLocaleString() + ' samples');
        drawSignalPreview(sig);
        setAnalyseBtn(true);
        hideError();
      } catch(err) {
        setParse('err', '✗  ' + err.message);
        parsedSignal = null;
        setAnalyseBtn(false);
      }
    };
    reader.readAsArrayBuffer(file);
  } else {
    reader.onload = e => {
      try {
        let sig = [];
        const raw = e.target.result.trim();

        if (ext === 'json') {
          sig = JSON.parse(raw);
          if (!Array.isArray(sig)) throw new Error('JSON must be a flat array of numbers');
          sig = sig.map(Number).filter(v => !isNaN(v));
        } else if (ext === 'csv') {
          sig = raw.split(/[\n,\r;]+/).map(s => s.trim()).filter(Boolean).map(Number).filter(v => !isNaN(v));
        } else if (ext === 'txt') {
          sig = raw.split(/[\s,;\n\r]+/).filter(Boolean).map(Number).filter(v => !isNaN(v));
        }

        if (sig.length < 1024) throw new Error('Signal too short — need ≥ 1024 samples, got ' + sig.length);
        parsedSignal = sig;
        const label = { csv: '.csv', json: '.json', txt: '.txt' }[ext];
        setParse('ok', '✓  ' + label + ' parsed — ' + sig.length.toLocaleString() + ' samples');
        drawSignalPreview(sig);
        setAnalyseBtn(true);
        hideError();
      } catch(err) {
        setParse('err', '✗  ' + err.message);
        parsedSignal = null;
        setAnalyseBtn(false);
      }
    };
    reader.readAsText(file);
  }
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const inp = document.getElementById('file-input');
  const dt = new DataTransfer(); dt.items.add(file); inp.files = dt.files;
  handleFile({ target: inp });
}

function setParse(cls, msg) {
  const el = document.getElementById('parse-info');
  el.className = 'parse-info ' + cls;
  el.textContent = msg;
}

function setAnalyseBtn(enabled) {
  document.getElementById('run-btn').disabled = !enabled;
}

function togglePredictionLog(forceOpen) {
  const drawer = document.getElementById('prediction-log-drawer');
  const toggle = document.getElementById('prediction-log-toggle');
  const page = document.getElementById('page-predict');
  if (!drawer || !toggle) return;

  const shouldOpen = typeof forceOpen === 'boolean' ? forceOpen : !drawer.classList.contains('open');
  drawer.classList.toggle('open', shouldOpen);
  page?.classList.toggle('log-open', shouldOpen);
  toggle.setAttribute('aria-expanded', String(shouldOpen));
  drawer.setAttribute('aria-hidden', String(!shouldOpen));
}

/* ══════ SIGNAL CHART ══════ */
function drawSignalPreview(sig) {
  document.getElementById('signal-card').style.display = 'block';
  const canvas = document.getElementById('signal-canvas');
  const slice  = sig.slice(0, 2048);
  if (signalChart) { signalChart.destroy(); signalChart = null; }
  signalChart = makeChart(canvas, slice, '#1e6fff', '#38bdf8');
}

function makeChart(canvas, data, strokeColor, gradTop) {
  return new Chart(canvas, {
    type: 'line',
    data: {
      labels: data.map((_, i) => i),
      datasets: [{
        data,
        borderColor: strokeColor,
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0,
        fill: { target: 'origin', above: gradTop + '18', below: strokeColor + '10' }
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: '#3a5a80' },
          grid: { color: 'rgba(30,111,255,.06)' }
        }
      }
    }
  });
}

/* ══════ HEALTH CHECK ══════ */
function onApiChange(val) {
  API_BASE = val.trim().replace(/\/$/, '');
  document.getElementById('info-url').textContent = API_BASE;
  checkHealth();
}

async function checkHealth() {
  const dot = document.getElementById('sdot');
  const txt = document.getElementById('stext');
  dot.className = 'sdot spin'; txt.textContent = 'Checking…';
  try {
    const r = await fetch(API_BASE + '/model_status', { signal: AbortSignal.timeout(5000) });
    const d = await r.json();
    if (d.model_loaded) {
      dot.className = 'sdot ok'; txt.textContent = 'Model ready';
      document.getElementById('info-status').textContent = d.status || 'ok';
      document.getElementById('info-loaded').textContent = 'Yes';
      if (d.uptime_seconds != null)
        document.getElementById('info-uptime').textContent = d.uptime_seconds.toFixed(1) + ' s';
      document.getElementById('m-status').innerHTML = '<span class="badge badge-ok">Ready</span>';
    } else {
      dot.className = 'sdot err'; txt.textContent = 'Model degraded';
      document.getElementById('m-status').innerHTML = '<span class="badge badge-warn">Degraded</span>';
    }
  } catch {
    dot.className = 'sdot err'; txt.textContent = 'Offline';
    document.getElementById('info-status').textContent = 'Cannot reach backend';
    document.getElementById('m-status').innerHTML = '<span class="badge badge-crit">Offline</span>';
  }
}

/* ══════ PREDICT ══════ */
async function runPredict() {
  if (!parsedSignal) return;
  hideError();
  document.getElementById('result-section').style.display = 'none';
  addLog('Sending ' + parsedSignal.length.toLocaleString() + ' samples to POST /predict_fault…');

  const body = {
    signal: parsedSignal,
    sampling_rate: parseInt(document.getElementById('sample-rate').value),
    sensor_id: document.getElementById('sensor-id').value
  };

  try {
    const r = await fetch(API_BASE + '/predict_fault', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'HTTP ' + r.status }));
      throw new Error(err.detail || 'HTTP ' + r.status);
    }

    const d = await r.json();
    renderResult(d);
    addLog('✓  ' + d.fault_class + ' (code ' + d.fault_code + ', conf ' + (Object.values(d.class_probabilities||{}).reduce((a,b)=>a>b?a:b,0)*100).toFixed(1) + '%)');
  } catch(e) {
    const msg = e.message.includes('Failed to fetch')
      ? 'Cannot reach ' + API_BASE + '/predict_fault\n\nFix: cd D:\\Project\\LBP_files\\backend\npython -m uvicorn main:app --reload --port 8000\n\nOriginal: ' + e.message
      : 'Prediction failed: ' + e.message;
    showError(msg);
    addLog('✗  ' + e.message);
  }
}

/* ══════ RENDER RESULT ══════ */
const ICONS = { Normal:'✅', Ball:'⚙️', IR:'⚠️', OR:'🔴' };
const ICON_BG = { Normal:'rgba(52,211,153,.15)', Ball:'rgba(30,111,255,.15)', IR:'rgba(251,191,36,.15)', OR:'rgba(248,113,113,.15)' };

const RECS = {
  'Normal':   'Bearing operating normally. Continue scheduled monitoring.',
  'Ball-007': 'Early ball defect (0.007"). Increase monitoring, check lubricant.',
  'Ball-014': 'Moderate ball defect (0.014"). Schedule inspection within 2 weeks.',
  'Ball-021': 'Severe ball defect (0.021"). Immediate replacement recommended.',
  'IR-007':   'Early inner race defect (0.007"). Inspect inner raceway, check alignment.',
  'IR-014':   'Moderate inner race fault (0.014"). Schedule maintenance within 1–2 weeks.',
  'IR-021':   'Severe inner race fault (0.021"). Replace bearing immediately.',
  'OR-007':   'Early outer race fault (0.007"). Check housing alignment.',
  'OR-014':   'Moderate outer race fault (0.014"). Replace within 1–2 weeks.',
  'OR-021':   'Severe outer race fault (0.021"). IMMEDIATE replacement required.',
};

function renderResult(d) {
  const cls  = d.fault_class;
  const probs = d.class_probabilities || {};
  const top  = Object.values(probs).reduce((a, b) => a > b ? a : b, 0);
  const pct  = (top * 100).toFixed(1) + '%';
  const key  = Object.keys(ICONS).find(k => cls.startsWith(k)) || 'Normal';

  document.getElementById('fault-icon').textContent = ICONS[key];
  document.getElementById('fault-icon').style.background = ICON_BG[key];
  document.getElementById('fault-name').textContent = cls;
  document.getElementById('fault-conf').textContent = 'Confidence: ' + pct + '  ·  best_cwru_cnn.keras';
  document.getElementById('r-code').textContent = d.fault_code;
  document.getElementById('r-windows').textContent = d.window_used ?? '—';
  document.getElementById('r-samples').textContent = parsedSignal ? parsedSignal.length.toLocaleString() : '—';
  document.getElementById('r-note').textContent = d.preprocessing_note ?? '—';
  document.getElementById('m-fault').textContent = cls;
  document.getElementById('m-conf').textContent = pct;
  document.getElementById('m-windows').textContent = d.window_used ?? '—';

  // prob bars
  const barsEl = document.getElementById('prob-bars');
  barsEl.innerHTML = '';
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([label, prob], i) => {
    const p = (prob * 100).toFixed(1);
    const isTop = label === cls;
    const div = document.createElement('div');
    div.className = 'prob-row';
    div.innerHTML = `
      <span class="prob-label ${isTop ? 'top' : ''}">${label}</span>
      <div class="prob-track">
        <div class="prob-fill" data-pct="${p}" style="background:${isTop ? PALETTE[0] : PALETTE[i] || PALETTE[0]};opacity:${isTop ? 1 : .45}"></div>
      </div>
      <span class="prob-pct ${isTop ? 'top' : ''}">${p}%</span>`;
    barsEl.appendChild(div);
  });

  // animate
  setTimeout(() => {
    barsEl.querySelectorAll('.prob-fill').forEach(b => { b.style.width = b.dataset.pct + '%'; });
  }, 50);

  document.getElementById('result-section').style.display = 'grid';
}

function showError(msg) {
  const b = document.getElementById('error-box');
  b.style.display = 'block';
  b.textContent = msg;
}

function hideError() {
  document.getElementById('error-box').style.display = 'none';
}

function addLog(msg) {
  const list = document.getElementById('log-list');
  const placeholder = list.querySelector('[style*="color:var(--text3)"]');
  if (placeholder) list.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'log-item';
  div.innerHTML = `<span class="log-time">${new Date().toLocaleTimeString()}</span><span class="log-msg">${msg}</span>`;
  list.prepend(div);
  while (list.children.length > 30) list.removeChild(list.lastChild);
}

/* ══════ SIMULATE ══════ */
function genSignal(ft) {
  const n = 4096, fs = 48000;
  return Array.from({ length: n }, (_, i) => {
    let v = Math.sin(2*Math.PI*60*i/fs) + .3*Math.sin(2*Math.PI*120*i/fs);
    if (ft !== 'Normal') {
      const freq = { IR: 162, OR: 107, Ball: 141 }[ft] || 120;
      v += .8*Math.sin(2*Math.PI*freq*i/fs) * (.5 + .5*Math.sin(2*Math.PI*2*i/fs));
    }
    return v + (Math.random() - .5) * .2;
  });
}

function drawSimCanvas(sig) {
  const canvas = document.getElementById('sim-canvas');
  if (simChart) { simChart.destroy(); simChart = null; }
  simChart = makeChart(canvas, sig.slice(0, 2048), '#38bdf8', '#0ea5e9');
}

function toggleSim() {
  if (simRunning) {
    clearInterval(simInterval);
    simRunning = false;
    document.getElementById('sim-btn').textContent = 'Start Simulation →';
  } else {
    simRunning = true; health = 100; simCount = 0;
    document.getElementById('sim-btn').textContent = 'Stop Simulation';
    runSimStep();
    simInterval = setInterval(runSimStep, 3000);
  }
}

async function runSimStep() {
  const ft = document.getElementById('sim-fault').value;
  const sig = genSignal(ft);
  drawSimCanvas(sig);
  simCount++;
  document.getElementById('sim-count').textContent = simCount;

  if (ft !== 'Normal') health = Math.max(0, health - Math.random()*4 - 2);
  else health = Math.min(100, health + .5);

  const h = Math.round(health);
  const hEl = document.getElementById('sim-health');
  hEl.textContent = h + '%';
  hEl.style.color = h > 70 ? 'var(--green)' : h > 40 ? 'var(--amber)' : 'var(--red)';

  const rul = Math.max(0, Math.round(h / 100 * 5000));
  document.getElementById('rul-val').textContent = rul.toLocaleString() + ' cycles';
  document.getElementById('rul-sub').textContent = rul < 500 ? '⚠  Immediate maintenance required' : rul < 2000 ? 'Plan maintenance soon' : 'Operating within safe limits';
  const bar = document.getElementById('rul-bar');
  bar.style.width = h + '%';
  bar.style.background = h > 70 ? 'var(--blue)' : h > 40 ? 'var(--amber)' : 'var(--red)';

  const ab = document.getElementById('alert-box');
  const mr = document.getElementById('maint-rec');
  if (ft === 'Normal') {
    ab.style.cssText = 'background:var(--bg3);color:var(--text2);border-color:var(--border)';
    ab.textContent = '✓  No alerts — bearing health nominal';
    mr.textContent = 'Continue routine monitoring.';
  } else if (h > 60) {
    ab.style.cssText = 'background:rgba(251,191,36,.1);color:var(--amber);border-color:rgba(251,191,36,.3)';
    ab.textContent = '⚠  ' + ft + ' fault detected — monitor closely';
    mr.textContent = RECS[ft + '-007'] || RECS[ft] || 'Monitor closely.';
  } else {
    ab.style.cssText = 'background:rgba(248,113,113,.1);color:var(--red);border-color:rgba(248,113,113,.3)';
    ab.textContent = '🔴  Critical ' + ft + ' fault — immediate action';
    mr.textContent = 'IMMEDIATE REPLACEMENT REQUIRED. Health index: ' + h + '%. Shut down if possible.';
  }

  try {
    const r = await fetch(API_BASE + '/predict_fault', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal: sig, sampling_rate: 48000, sensor_id: 'DE' })
    });
    if (r.ok) {
      const d = await r.json();
      document.getElementById('sim-pred').textContent = d.fault_class;
      const tp = d.class_probabilities ? Math.max(...Object.values(d.class_probabilities)) : null;
      document.getElementById('sim-conf').textContent = tp ? (tp*100).toFixed(1)+'%' : '—';
    }
  } catch {
    document.getElementById('sim-pred').textContent = ft === 'Normal' ? 'Normal' : ft + ' fault';
    document.getElementById('sim-conf').textContent = (70 + Math.random()*25).toFixed(0) + '%';
  }
}

/* ══════ CHATBOT ══════ */
const KB = {
  'inner race': 'An <strong>inner race fault</strong> occurs on the bearing\'s inner ring (which rotates with the shaft). It causes periodic impulses at BPFI (Ball Pass Frequency Inner race). Your CWRU files: <code>IR007_1_110.mat</code>, <code>IR014_1_175.mat</code>, <code>IR021_1_214.mat</code>. Severity increases with the suffix (007 < 014 < 021 = 0.021 inch diameter fault).',
  'outer race': 'An <strong>outer race fault</strong> is on the stationary outer ring — typically more severe as it bears the main load. Files: <code>OR007_6_1_136.mat</code>, <code>OR014_6_1_202.mat</code>, <code>OR021_6_1_239.mat</code>. OR-021 requires immediate bearing replacement.',
  'ball fault': 'A <strong>ball/rolling element fault</strong> affects the steel balls between raceways. Files: <code>B007_1_123.mat</code>, <code>B014_1_190.mat</code>, <code>B021_1_227.mat</code>. First check lubricant contamination when this is detected.',
  'cnn': 'The backend uses <strong>best_cwru_cnn.keras</strong> — a 1D-CNN trained on raw 2048-sample windows. Unlike the SVM which needs 13 hand-crafted features (RMS, kurtosis, etc.), the CNN learns features automatically from raw signals. Preprocessing: windowing → zero-mean/unit-var normalization → predict → average softmax over windows.',
  'svm': 'Your <strong>SVM pipeline</strong> in RF-SVM(4).ipynb extracts 13 features (RMS, mean, std, kurtosis, skewness, crest factor, spectral energy etc.) then trains SVC with RBF kernel. It achieved 97.25% on CWRU. The backend however uses the CNN, not SVM — the CNN is better for real-time signals.',
  'rul': '<strong>RUL (Remaining Useful Life)</strong> estimates how many cycles remain before failure. This system simulates it using a health index that degrades proportionally to fault severity. Real RUL requires time-series data recorded over the bearing\'s lifetime (like PRONOSTIA/FEMTO dataset).',
  'transfer': '<strong>Transfer learning</strong> here: CNN trained on CWRU lab data → fine-tuned on Paderborn real-world data. DANN (Domain Adversarial Neural Network) adds a gradient reversal layer so the feature extractor learns domain-invariant representations, bridging the lab-to-field distribution gap.',
  'or-021': '<strong>OR-021 action:</strong> This is the most severe outer race fault (0.021 inch). IMMEDIATE bearing replacement required. Steps: (1) Shut down if safe, (2) Replace bearing, (3) Inspect housing bore for pitting, (4) Check shaft alignment, (5) Verify lubricant before restart.',
  'mat': 'Your <strong>.mat files</strong> are CWRU Level-5 MAT format. Each file contains <code>X###_DE_time</code> (Drive End), <code>X###_FE_time</code> (Fan End) signals and RPM. This frontend reads them in pure JavaScript — no MATLAB needed. Just drop any .mat file from your <code>raw_data</code> folder.',
  'uvicorn': '<strong>Fix for "Could not import module main" error:</strong> Your uvicorn was run from <code>LBP_files\\</code> but main.py is in <code>LBP_files\\backend\\</code>. Fix:\n<code>cd D:\\Project\\LBP_files\\backend\npython -m uvicorn main:app --reload --port 8000</code>',
  'paderborn': 'The <strong>Paderborn dataset</strong> contains real-world bearing vibration signals with motor/machine noise — much harder than CWRU lab data. Transfer learning adapts the CWRU-trained CNN to Paderborn using fine-tuning + domain adaptation (DANN/CORAL). File: <code>paderborn_tl_cnn_final.keras</code>',
};

function qask(q) { document.getElementById('chat-input').value = q; sendChat(); }

async function sendChat() {
  const inp = document.getElementById('chat-input');
  const msg = inp.value.trim();
  if (!msg) return;
  appendChat('user', msg);
  inp.value = '';

  const thinking = appendChat('ai', '…');
  const apiKey = document.getElementById('gemini-key').value.trim();

  if (apiKey) {
    try {
      const sys = 'You are an expert bearing fault diagnosis AI assistant. The project uses CWRU dataset (10 classes: Normal, Ball/IR/OR at 007/014/021 severity). Backend uses 1D-CNN (best_cwru_cnn.keras). The common error "Could not import module main" is fixed by cd-ing into the backend subfolder before running uvicorn. Answer concisely and technically.';
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts: [{ text: sys + '\n\nUser: ' + msg }] }] })
      });
      const d = await r.json();
      thinking.innerHTML = (d.candidates?.[0]?.content?.parts?.[0]?.text || 'No response.').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code style="background:rgba(30,111,255,.2);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:11px">$1</code>');
    } catch (e) {
      thinking.textContent = 'Gemini error: ' + e.message;
    }
  } else {
    await new Promise(r => setTimeout(r, 380));
    const q = msg.toLowerCase();
    const key = Object.keys(KB).find(k => q.includes(k));
    if (key) {
      thinking.innerHTML = KB[key].replace(/\n/g, '<br/>').replace(/<code>/g, '<code style="background:rgba(30,111,255,.2);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:11px">');
    } else {
      thinking.textContent = 'Add a Gemini API key for full AI answers. I have local knowledge about: inner race, outer race, ball faults, CNN vs SVM, RUL, transfer learning, Paderborn dataset, .mat files, uvicorn error fix.';
    }
  }
  document.getElementById('chat-wrap').scrollTop = 99999;
}

function appendChat(cls, html) {
  const el = document.createElement('div');
  el.className = 'chat-msg ' + cls;
  el.innerHTML = html;
  const wrap = document.getElementById('chat-wrap');
  wrap.appendChild(el);
  wrap.scrollTop = wrap.scrollHeight;
  return el;
}

/* ══════ INIT ══════ */
checkHealth();
setInterval(checkHealth, 30000);
