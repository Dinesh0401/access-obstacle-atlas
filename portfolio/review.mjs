/** Every overlay is a fraction of the very same oriented bitmap fed to the detector. */
const el = id => document.getElementById(id);
const reviewInputs = [...document.querySelectorAll('input[name="verdict"]')];
let revision = 0;
let active = null;
let drag = null;

function status(message) {
  el('status').textContent = message;
}

function clearDesk() {
  active = null;
  drag = null;
  el('manual-pane').hidden = true;
  el('manual-pane').open = false;
  el('manual-form').reset();
  el('manual-message').textContent = '';
  el('draft').hidden = true;
  el('boxes').replaceChildren();
  el('object-list').replaceChildren();
  el('review-tools').hidden = true;
  el('retry').hidden = true;
  el('plane').hidden = true;
  el('photo-credit').hidden = true;
  el('dimensions').textContent = '';
}

async function fingerprint(blob) {
  if (!globalThis.crypto?.subtle) return null;
  const bytes = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(bytes)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

async function orientedCanvas(blob) {
  if (!blob.size || blob.size > 16 * 1024 * 1024) throw new Error('Choose a photo under 16 MiB.');
  const bitmap = await createImageBitmap(blob, {imageOrientation:'from-image'});
  try {
    if (bitmap.width * bitmap.height > 48_000_000) throw new Error('Choose a photo under 48 megapixels.');
    const scale = Math.min(1, 1440 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas;
  } finally {
    bitmap.close();
  }
}

function validDetections(raw) {
  if (!Array.isArray(raw)) throw new Error('The detector returned an unreadable result.');
  return raw.flatMap(item => {
    if (typeof item.label !== 'string' || !Number.isFinite(item.score) || !Array.isArray(item.box) ||
        item.box.length !== 4 || !item.box.every(Number.isFinite)) return [];
    const box = item.box.map(value => Math.min(1, Math.max(0, value)));
    if (box[2] <= box[0] || box[3] <= box[1]) return [];
    return [{...item, box}];
  });
}

function observationKey(item) {
  return JSON.stringify([...(item.manual ? ['human'] : []), item.label, ...item.box.map(value => Number(value.toFixed(4)))]);
}

function selectObject(index) {
  if (!active || !active.objects[index]) return;
  active.selection = index;
  el('object-list').selectedIndex = index;
  [...el('boxes').children].forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
  const item = active.objects[index];
  const [left,top,right,bottom] = item.box.map(value => Math.round(100*value));
  el('position').textContent = `Image bounds: ${left}–${right}% from left, ${top}–${bottom}% from top.`;
  const choice = active.votes[observationKey(item)];
  for (const input of reviewInputs) input.checked = input.value === choice;
  const done = active.objects.filter(obj => active.votes[observationKey(obj)]).length;
  el('progress').textContent = `${done} of ${active.objects.length} objects reviewed.`;
}

function renderObjects(scene) {
  el('boxes').replaceChildren();
  el('object-list').replaceChildren();
  if (!scene.objects.length) {
    status('No objects detected. Inspect the photo yourself; the walking surface may still be obstructed.');
    return;
  }
  scene.objects.forEach((obj, index) => {
    const [left,top,right,bottom] = obj.box;
    const source = obj.manual ? 'human-marked region' : Math.round(obj.score*100) + '% detector score';
    const title = `${index+1}. ${obj.label} · ${source}`;
    el('object-list').add(new Option(title, String(index)));
    const button = document.createElement('button');
    button.type = 'button';
    button.className = obj.manual ? 'box manual-box' : 'box';
    button.setAttribute('aria-label', title);
    button.title = title;
    Object.assign(button.style, {left:`${100*left}%`,top:`${100*top}%`,width:`${100*(right-left)}%`,height:`${100*(bottom-top)}%`});
    const label = document.createElement('span');
    label.textContent = String(index+1);
    button.append(label);
    button.addEventListener('click', () => selectObject(index));
    el('boxes').append(button);
  });
  el('review-tools').hidden = false;
  const found = scene.objects.filter(obj => !obj.manual).length;
  status(`${found} detected · ${scene.manual.length} human-marked. Select a region to review.`);
  selectObject(0);
}

async function inspect(scene) {
  if (scene !== active || scene.running) return;
  scene.running = true;
  el('retry').hidden = true;
  status('Analyzing the photograph… The first run can take longer while the detector loads.');
  try {
    const {detect} = await import('./detector.mjs');
    const output = await detect(scene.canvas);
    if (active !== scene) return;
    scene.objects = [...validDetections(output), ...scene.manual];
    renderObjects(scene);
  } catch (error) {
    if (active !== scene) return;
    console.error('Object detection failed', error);
    status('Detection could not finish. The photo is available for inspection. Check the connection and retry.');
    el('retry').hidden = false;
  } finally {
    scene.running = false;
  }
}

async function openPhoto(file = null) {
  const version = ++revision;
  clearDesk();
  const name = file ? file.name : 'Tactile paving with obstructions';
  el('photo-name').textContent = name;
  status('Opening photograph…');
  try {
    let blob = file;
    if (!blob) {
      const response = await fetch('./field-example.jpg');
      if (!response.ok) throw new Error('The field photograph could not be loaded.');
      blob = await response.blob();
    } else if (!['image/jpeg','image/png','image/webp'].includes(blob.type.toLowerCase())) {
      throw new Error('Choose a JPEG, PNG or WebP photograph.');
    }
    const [canvas, digest] = await Promise.all([orientedCanvas(blob),fingerprint(blob)]);
    if (version !== revision) return;
    const storageKey = digest ? 'atlas:object-review:v1:' + digest : null;
    let votes = {};
    try {
      const stored = storageKey && JSON.parse(localStorage.getItem(storageKey));
      if (stored && typeof stored === 'object' && !Array.isArray(stored)) votes = stored;
    } catch { /* Private sessions can restrict storage. Reviews remain in memory. */ }
    canvas.id = 'source';
    canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label', file ? 'Uploaded street photograph: ' + file.name : 'A parked scooter beside tactile paving');
    el('source').replaceWith(canvas);
    const plane = el('plane');
    plane.style.setProperty('--natural-limit', `${620*canvas.width/canvas.height}px`);
    plane.hidden = false;
    el('dimensions').textContent = `${canvas.width} × ${canvas.height} inspection pixels`;
    el('photo-credit').hidden = Boolean(file);
    let manual = [];
    try {
      const stored = storageKey && JSON.parse(localStorage.getItem(storageKey + ':regions'));
      if (Array.isArray(stored)) manual = stored.filter(validManual).map(({label,box}) => ({label,box,manual:true}));
    } catch { /* An unavailable or outdated cache is optional. */ }
    const scene = {canvas, storageKey, votes, manual, objects:[...manual], selection:0, running:false};
    active = scene;
    el('manual-pane').hidden = false;
    if (manual.length) renderObjects(scene);
    await inspect(scene);
  } catch (error) {
    if (version !== revision) return;
    status(error instanceof Error && error.name === 'Error' ? error.message : 'This photo could not be read. Choose another JPEG, PNG or WebP file.');
  }
}

const edges = ['left', 'top', 'right', 'bottom'];

function validManual(region) {
  return region && typeof region.label === 'string' && region.label.trim().length > 0 &&
    region.label.length <= 60 && Array.isArray(region.box) && region.box.length === 4 &&
    region.box.every(v => Number.isFinite(v) && v >= 0 && v <= 1) &&
    region.box[2] > region.box[0] && region.box[3] > region.box[1];
}

function draftBox() {
  const box = edges.map(id => el(id).valueAsNumber/100);
  const preview = el('draft');
  preview.hidden = !active || !el('manual-pane').open || !validManual({label:'draft',box});
  if (!preview.hidden) {
    const [l,t,r,b] = box;
    Object.assign(preview.style, {left:l*100+'%',top:t*100+'%',width:(r-l)*100+'%',height:(b-t)*100+'%'});
  }
  return box;
}

function point(event) {
  const bounds = el('source').getBoundingClientRect();
  const percentage = fraction => Math.round(Math.min(1,Math.max(0,fraction))*1000)/10;
  return [percentage((event.clientX-bounds.left)/bounds.width), percentage((event.clientY-bounds.top)/bounds.height)];
}

function updateDrag(event) {
  if (!drag || drag.id !== event.pointerId) return;
  const [x,y] = point(event);
  const [x0,y0] = drag.start;
  const box = [Math.min(x,x0),Math.min(y,y0),Math.max(x,x0),Math.max(y,y0)];
  edges.forEach((edge,i) => {el(edge).value = box[i];});
  draftBox();
}

el('manual-pane').addEventListener('toggle', () => {
  el('plane').classList.toggle('marking',el('manual-pane').open);
  drag = null;
  draftBox();
});
for (const edge of edges) el(edge).addEventListener('input',draftBox);
el('plane').addEventListener('pointerdown', event => {
  if (!active || !el('manual-pane').open || event.button !== 0) return;
  drag = {id:event.pointerId,start:point(event)};
  el('plane').setPointerCapture(event.pointerId);
  updateDrag(event);
});
el('plane').addEventListener('pointermove', updateDrag);
el('plane').addEventListener('pointerup', event => {updateDrag(event); drag = null;});
el('plane').addEventListener('pointercancel', () => {drag = null;});
el('manual-form').addEventListener('submit', event => {
  event.preventDefault();
  const scene = active;
  if (!scene) return;
  const region = {label:el('manual-name').value.trim(), box:draftBox(), manual:true};
  if (!validManual(region)) {
    el('manual-message').textContent = 'Enter an obstacle name and bounds with right greater than left and bottom greater than top.';
    return;
  }
  if (!scene.manual.some(item => observationKey(item) === observationKey(region))) {
    scene.manual.push(region);
    scene.objects.push(region);
  }
  let saved = true;
  try {
    if (!scene.storageKey) throw new Error('Storage unavailable');
    localStorage.setItem(scene.storageKey + ':regions', JSON.stringify(scene.manual));
  } catch { saved = false; }
  renderObjects(scene);
  selectObject(scene.objects.findIndex(obj => observationKey(obj) === observationKey(region)));
  el('manual-form').reset();
  el('manual-pane').open = false;
  el('draft').hidden = true;
  el('manual-message').textContent = '';
  el('progress').textContent += saved ? ' Human-marked region saved on this device.' : ' Human-marked region available for this visit.';
  el('object-list').focus();
});

el('upload').addEventListener('click', () => el('photo').click());
el('photo').addEventListener('change', event => {
  const [file] = event.target.files;
  event.target.value = '';
  if (file) openPhoto(file);
});
el('field').addEventListener('click', () => openPhoto());
el('retry').addEventListener('click', () => active && inspect(active));
el('overlay').addEventListener('change', event => {el('boxes').hidden = !event.target.checked;});
el('object-list').addEventListener('change', event => selectObject(event.target.selectedIndex));
for (const input of reviewInputs) {
  input.addEventListener('change', () => {
    const scene = active;
    if (!scene || !input.checked || !scene.objects[scene.selection]) return;
    scene.votes[observationKey(scene.objects[scene.selection])] = input.value;
    selectObject(scene.selection);
    try {
      if (!scene.storageKey) throw new Error('Storage unavailable');
      localStorage.setItem(scene.storageKey, JSON.stringify(scene.votes));
    } catch {
      el('progress').textContent += ' Browser storage is unavailable; this review lasts for this visit.';
    }
  });
}
openPhoto();
