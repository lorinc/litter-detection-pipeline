// Browser port of the pipeline's YOLO scoring (pipeline/detectors.py YOLO + pipeline/tiling.py, fit=True):
// 640 px square tiles at 384 px spacing, edge tiles shifted inward, per-tile NMS at IoU 0.7 (Ultralytics' default),
// boxes clipped to the tile, mapped back to the photo and merged with NMS at IoU 0.3. Inputs are RGB / 255; a tile
// beyond a small photo's edge is padded with grey 114, as Ultralytics' letterbox does.
const TILE = 640, STRIDE = 384, RAW = 0.01, NMS_TILE = 0.7, NMS_ALL = 0.3, MAX_DET = 300;
// All three models train on the same 87 photos. Keys are initiative 12's run names; labels and info are what visitors
// see (info is shown under the selector).
// Every detection down to RAW is shown, coloured by its score band: [lowest score, box colour, label text colour].
const BANDS = [[0.78, '#34c759', '#1f1f1d'], [0.65, '#ff9500', '#1f1f1d'], [0, '#ff3b30', '#fff']];
const band = s => BANDS.find(b => s >= b[0]);
const MODELS = {
  '02-maybe-masked':         { file: 'model/yolo11s-02-maybe-masked.onnx',         label: 'YOLO11s first',    mb: 38,
    info: 'Standard-size model (9.4 M parameters, 38 MB download), fine-tuned on 87 photos from one walk, without the look-alike spots. More false alarms than YOLO11s adjusted.' },
  '04-cv-hard-negatives-f0': { file: 'model/yolo11s-04-cv-hard-negatives-f0.onnx', label: 'YOLO11s adjusted', mb: 38,
    info: 'Standard-size model (9.4 M parameters, 38 MB download), fine-tuned on 87 photos from one walk. Adjusted: also trained on spots that fool OWLv2, an off-the-shelf, generic vision model (cracks, gum, leaves), so it raises fewer false alarms.' },
  '06-cv-yolo11n-f0':        { file: 'model/yolo11n-06-cv-yolo11n-f0.onnx',        label: 'YOLO11n fast',     mb: 11,
    info: 'Nano model, a quarter of the size (2.6 M parameters, 11 MB download), with the same training as YOLO11s adjusted. About 3× faster; misses a few more butts.' },
};
const DEFAULT_MODEL = '04-cv-hard-negatives-f0';
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const EXAMPLES = [
  ['20261001_134106', 'wide, shade, square tiles and manholes'],
  ['20261001_134246', 'wide, mixed light, red brick and asphalt'],
  ['20261001_134457', 'wide, sun, granite step and kerb'],
  ['20261001_135420', 'high angle, shade, tactile tiles'],
];

const $ = id => document.getElementById(id);
const CPU_NOTE = 'This browser offers no WebGPU, so detection runs on the CPU: expect about a minute per full-resolution photo. Chrome and Edge on Windows, macOS and ChromeOS run it on the GPU by default. In Chrome on Linux, turn on chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, restart the browser and reload this page.';
const PARAMS = new URLSearchParams(location.search);   // ?cpu=1 skips WebGPU; ?example=STEM runs one example at load
const sessions = {};                                // model key -> InferenceSession
let backend = '', current = null, runId = 0;         // current = {bitmap, dets, name}
let modelKey = MODELS[PARAMS.get('model')] ? PARAMS.get('model') : DEFAULT_MODEL;   // ?model=06-cv-yolo11n-f0 picks run 06

function starts(n) {
  if (n <= TILE) return [0];
  const s = [];
  for (let x = 0; x <= n - TILE; x += STRIDE) s.push(x);
  if (s[s.length - 1] + TILE < n) s.push(n - TILE);
  return s;
}

function nms(dets, iou) {                          // dets: [score, x0, y0, x1, y1], returns kept, by score
  dets.sort((a, b) => b[0] - a[0]);
  const keep = [], dead = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length; i++) {
    if (dead[i]) continue;
    const a = dets[i]; keep.push(a);
    const areaA = (a[3] - a[1]) * (a[4] - a[2]);
    for (let j = i + 1; j < dets.length; j++) {
      if (dead[j]) continue;
      const b = dets[j];
      const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]), h = Math.min(a[4], b[4]) - Math.max(a[2], b[2]);
      if (w <= 0 || h <= 0) continue;
      const inter = w * h;
      if (inter / (areaA + (b[3] - b[1]) * (b[4] - b[2]) - inter) > iou) dead[j] = 1;
    }
  }
  return keep;
}

async function loadModel(key) {
  if (sessions[key]) return sessions[key];
  let session = null;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  status(`Loading the model "${MODELS[key].label}" (${MODELS[key].mb} MB, once)…`);
  const bytes = new Uint8Array(await (await fetch(MODELS[key].file)).arrayBuffer());
  if (navigator.gpu && !PARAMS.has('cpu')) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] }); backend = 'GPU (WebGPU)'; }
    catch (e) { console.warn('WebGPU unavailable, using CPU:', e); }
  }
  if (!session) {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }); backend = 'CPU (WebAssembly)';
    if (!$('warn').textContent.includes(CPU_NOTE)) {
      $('warn').style.display = 'block'; $('warn').textContent = [CPU_NOTE, $('warn').textContent].filter(Boolean).join(' ');
    }
  }
  return sessions[key] = session;
}

async function detect(bitmap, session, id) {   // returns null when a newer run has started
  const W = bitmap.width, H = bitmap.height;
  const xs = starts(W), ys = starts(H), n = xs.length * ys.length;
  const cv = new OffscreenCanvas(TILE, TILE), cx = cv.getContext('2d', { willReadFrequently: true });
  const input = new Float32Array(3 * TILE * TILE), P = TILE * TILE;
  const all = []; let k = 0; const t0 = performance.now();
  for (const ty of ys) for (const tx of xs) {
    if (id !== runId) return null;
    k++; status(`Detecting with "${MODELS[modelKey].label}" on ${backend}: tile ${k} of ${n}…`);
    cx.fillStyle = 'rgb(114,114,114)'; cx.fillRect(0, 0, TILE, TILE);
    const tw = Math.min(TILE, W - tx), th = Math.min(TILE, H - ty);
    cx.drawImage(bitmap, tx, ty, tw, th, 0, 0, tw, th);
    const px = cx.getImageData(0, 0, TILE, TILE).data;
    for (let i = 0; i < P; i++) { input[i] = px[4 * i] / 255; input[P + i] = px[4 * i + 1] / 255; input[2 * P + i] = px[4 * i + 2] / 255; }
    const out = (await session.run({ images: new ort.Tensor('float32', input, [1, 3, TILE, TILE]) })).output0;
    const d = out.data, A = out.dims[2];          // [1, 5, A]: cx, cy, w, h, score
    const tile = [];
    for (let i = 0; i < A; i++) {
      const s = d[4 * A + i]; if (s < RAW) continue;
      const bx = d[i], by = d[A + i], bw = d[2 * A + i], bh = d[3 * A + i];
      tile.push([s, bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2]);
    }
    const cl = (v, hi) => Math.min(Math.max(v, 0), hi);   // Ultralytics clips each box to the tile
    for (const b of nms(tile, NMS_TILE).slice(0, MAX_DET))
      all.push([b[0], cl(b[1], tw) + tx, cl(b[2], th) + ty, cl(b[3], tw) + tx, cl(b[4], th) + ty]);
    await new Promise(r => setTimeout(r));          // let the page repaint the progress line
  }
  return { dets: nms(all, NMS_ALL), ms: performance.now() - t0, tiles: n };
}

function draw() {
  const c = $('view'), g = c.getContext('2d');
  if (!current) return;
  const { bitmap, dets } = current;
  c.width = bitmap.width; c.height = bitmap.height;
  g.drawImage(bitmap, 0, 0);
  if (!dets) return;
  const lw = Math.max(3, bitmap.width / 700), fs = Math.max(22, bitmap.width / 110);
  g.lineWidth = lw; g.font = `600 ${fs}px system-ui, sans-serif`; g.textBaseline = 'bottom';
  for (const [s, x0, y0, x1, y1] of [...dets].reverse()) {   // lowest scores first, so higher bands draw on top
    const [, col, ink] = band(s), pad = lw * 2;
    g.strokeStyle = col; g.strokeRect(x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad);
    const label = Math.round(s * 100) + '%', tw = g.measureText(label).width;
    const ly = y0 - pad - fs - 4 >= 0 ? y0 - pad - fs - 4 : y1 + pad;   // label above the box, or below it at the top edge
    g.fillStyle = col; g.fillRect(x0 - pad, ly, tw + 10, fs + 4);
    g.fillStyle = ink; g.fillText(label, x0 - pad + 5, ly + fs + 2);
  }
  drawLoupes();
}

// Hover magnifiers, as in initiative 12's annotator: one column of square loupes at fixed zooms of the photo's own
// pixels, on the side of the window away from the cursor, with thin detection boxes and a crosshair.
const LOUPE_ZOOMS = [2, 6, 12], LOUPE_PX = 220;
const loupes = LOUPE_ZOOMS.map(z => {
  const c = document.createElement('canvas'); $('loupes').appendChild(c); return { z, c, g: c.getContext('2d') };
});
let mouse = null;                                    // pointer position over the photo, client coordinates

function drawLoupes() {
  const box = $('loupes');
  if (!mouse || !current) { box.style.display = 'none'; return; }
  const { bitmap, dets } = current, r = $('view').getBoundingClientRect();
  const nx = (mouse.x - r.left) * bitmap.width / r.width, ny = (mouse.y - r.top) * bitmap.height / r.height;
  const L = Math.floor(Math.min(LOUPE_PX, (innerHeight - 24 - 6 * (loupes.length - 1)) / loupes.length));
  box.style.display = 'flex'; box.style.left = mouse.x < innerWidth / 2 ? (innerWidth - L - 12) + 'px' : '12px';
  for (const { z, c, g } of loupes) {
    if (c.width !== L) c.width = c.height = L;
    g.imageSmoothingEnabled = false; g.fillStyle = '#000'; g.fillRect(0, 0, L, L);
    const sw = L / z; g.drawImage(bitmap, nx - sw / 2, ny - sw / 2, sw, sw, 0, 0, L, L);
    const X = x => (x - nx) * z + L / 2, Y = y => (y - ny) * z + L / 2;
    g.lineWidth = 1.5;
    for (const [s, x0, y0, x1, y1] of [...(dets || [])].reverse()) { g.strokeStyle = band(s)[1]; g.strokeRect(X(x0), Y(y0), (x1 - x0) * z, (y1 - y0) * z); }
    g.strokeStyle = 'rgba(0,255,255,.8)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(L / 2, 0); g.lineTo(L / 2, L); g.moveTo(0, L / 2); g.lineTo(L, L / 2); g.stroke();
    g.fillStyle = '#0ff'; g.font = '12px sans-serif'; g.fillText(z + '×', 5, 15);
  }
}

function status(t) { $('status').textContent = t; }

function warnFor(bitmap) {
  const W = bitmap.width, H = bitmap.height, notes = [];
  if (backend.startsWith('CPU')) notes.push(CPU_NOTE);
  if (Math.max(W, H) < 3000) notes.push(`This photo is ${W}×${H} px. The model was trained on full-resolution photos (about 4000×1800 px); in a smaller copy every butt covers fewer pixels and small ones are missed. Use the original file from the camera if you can.`);
  if (H > W) notes.push('This photo is portrait. The model was trained mostly on landscape photos of the ground ahead.');
  $('warn').style.display = notes.length ? 'block' : 'none';
  $('warn').textContent = notes.join(' ');
}

async function run(bitmap, name) {
  const id = ++runId, key = modelKey;
  current = { bitmap, dets: null, name }; warnFor(bitmap); draw();
  try {
    const session = await loadModel(key);
    if (id !== runId) return;
    const r = await detect(bitmap, session, id);
    if (!r) return;
    current.dets = r.dets; draw();
    const n = BANDS.map(([lo], i) => r.dets.filter(d => d[0] >= lo && (i === 0 || d[0] < BANDS[i - 1][0])).length);
    status(`${name}: ${r.dets.length} detections with "${MODELS[key].label}" (${n[0]} at 78% and up, ${n[1]} at 65–78%, ${n[2]} below 65%) · ${r.tiles} tiles in ${(r.ms / 1000).toFixed(1)} s on ${backend}.`);
    window.lastResult = { name, model: key, backend, ms: r.ms, tiles: r.tiles, dets: r.dets };   // read by the page test
  } catch (e) { console.error(e); if (id === runId) status('Detection failed: ' + e.message); }
}

function showModel() {
  const m = MODELS[modelKey];
  $('modelinfo').innerHTML = `<strong>${m.label}</strong>: ${m.info}`;
}

async function runExample(stem) {
  status('Loading the example photo…');
  const blob = await (await fetch(`examples/${stem}.jpg`)).blob();
  run(await createImageBitmap(blob), `Example ${stem}`);
}

for (const [stem, text] of EXAMPLES) {
  const b = document.createElement('button');
  b.innerHTML = `<img src="examples/${stem}_thumb.jpg" alt="Example photo: ${text}"><span>${text}</span>`;
  b.onclick = () => runExample(stem);
  $('gallery').appendChild(b);
}
$('file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { run(await createImageBitmap(f, { imageOrientation: 'from-image' }), f.name); }
  catch (err) { status('This file could not be read as an image.'); }
};
for (const [key, m] of Object.entries(MODELS)) {
  const o = document.createElement('option'); o.value = key; o.textContent = m.label; $('model').appendChild(o);
}
$('model').value = modelKey; showModel();
$('model').onchange = () => {
  modelKey = $('model').value; showModel();
  if (current) run(current.bitmap, current.name);   // switching re-runs detection on the current photo
};
$('view').addEventListener('pointermove', e => {
  mouse = e.pointerType === 'mouse' ? { x: e.clientX, y: e.clientY } : null; drawLoupes();
});
$('view').addEventListener('pointerleave', () => { mouse = null; drawLoupes(); });
addEventListener('scroll', () => { mouse = null; drawLoupes(); }, { passive: true });   // the photo moved under a still cursor
if (PARAMS.get('example')) runExample(PARAMS.get('example'));
