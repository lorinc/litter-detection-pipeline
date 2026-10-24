// Browser port of the pipeline's YOLO scoring (pipeline/detectors.py YOLO + pipeline/tiling.py, fit=True):
// 640 px square tiles at 384 px spacing, edge tiles shifted inward, per-tile NMS at IoU 0.7 (Ultralytics' default),
// boxes mapped back to the photo and merged with NMS at IoU 0.3. Inputs are RGB / 255; a tile beyond a small
// photo's edge is padded with grey 114, as Ultralytics' letterbox does.
const TILE = 640, STRIDE = 384, RAW = 0.01, NMS_TILE = 0.7, NMS_ALL = 0.3, MAX_DET = 300;
const MODEL = 'model/yolo11s-run2.onnx';
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
let session = null, backend = '', current = null;   // current = {bitmap, dets, name}

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

async function loadModel() {
  if (session) return session;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  status('Loading the model (38 MB, once)…');
  const bytes = new Uint8Array(await (await fetch(MODEL)).arrayBuffer());
  if (navigator.gpu && !PARAMS.has('cpu')) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] }); backend = 'GPU (WebGPU)'; }
    catch (e) { console.warn('WebGPU unavailable, using CPU:', e); }
  }
  if (!session) {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }); backend = 'CPU (WebAssembly)';
    $('warn').style.display = 'block'; $('warn').textContent = [CPU_NOTE, $('warn').textContent].filter(Boolean).join(' ');
  }
  return session;
}

async function detect(bitmap) {
  const W = bitmap.width, H = bitmap.height;
  const xs = starts(W), ys = starts(H), n = xs.length * ys.length;
  const cv = new OffscreenCanvas(TILE, TILE), cx = cv.getContext('2d', { willReadFrequently: true });
  const input = new Float32Array(3 * TILE * TILE), P = TILE * TILE;
  const all = []; let k = 0; const t0 = performance.now();
  for (const ty of ys) for (const tx of xs) {
    k++; status(`Detecting on ${backend}: tile ${k} of ${n}…`);
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
    for (const b of nms(tile, NMS_TILE).slice(0, MAX_DET)) all.push([b[0], b[1] + tx, b[2] + ty, b[3] + tx, b[4] + ty]);
    await new Promise(r => setTimeout(r));          // let the page repaint the progress line
  }
  return { dets: nms(all, NMS_ALL), ms: performance.now() - t0, tiles: n };
}

function draw() {
  const c = $('view'), g = c.getContext('2d');
  if (!current) return;
  const { bitmap, dets } = current, thr = +$('thr').value;
  c.width = bitmap.width; c.height = bitmap.height;
  g.drawImage(bitmap, 0, 0);
  if (!dets) return;
  const lw = Math.max(3, bitmap.width / 700), fs = Math.max(22, bitmap.width / 110);
  g.lineWidth = lw; g.font = `600 ${fs}px system-ui, sans-serif`; g.textBaseline = 'bottom';
  let shown = 0;
  for (const [s, x0, y0, x1, y1] of dets) {
    if (s < thr) continue; shown++;
    const pad = lw * 2;
    g.strokeStyle = '#ff3b30'; g.strokeRect(x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad);
    const label = s.toFixed(2), tw = g.measureText(label).width;
    const ly = y0 - pad - fs - 4 >= 0 ? y0 - pad - fs - 4 : y1 + pad;   // label above the box, or below it at the top edge
    g.fillStyle = '#ff3b30'; g.fillRect(x0 - pad, ly, tw + 10, fs + 4);
    g.fillStyle = '#fff'; g.fillText(label, x0 - pad + 5, ly + fs + 2);
  }
  current.shown = shown;
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
  current = { bitmap, dets: null, name }; warnFor(bitmap); draw();
  try {
    await loadModel();
    const r = await detect(bitmap);
    current.dets = r.dets; draw();
    status(`${name}: ${current.shown} detections at threshold ${(+$('thr').value).toFixed(2)} · ${r.tiles} tiles in ${(r.ms / 1000).toFixed(1)} s on ${backend}.`);
    window.lastResult = { name, backend, ms: r.ms, tiles: r.tiles, dets: r.dets };   // read by the page test
  } catch (e) { console.error(e); status('Detection failed: ' + e.message); }
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
$('thr').oninput = () => {
  $('thrv').textContent = (+$('thr').value).toFixed(2); draw();
  if (current && current.dets) status(`${current.name}: ${current.shown} detections at threshold ${(+$('thr').value).toFixed(2)} on ${backend}.`);
};
if (PARAMS.get('example')) runExample(PARAMS.get('example'));
