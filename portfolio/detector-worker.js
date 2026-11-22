/* CPU-only TensorFlow.js COCO-SSD inference. No image or URL leaves the worker. */
importScripts('./detector-assets/tf.min.js', './detector-assets/coco-ssd.min.js');
const fieldObjects = new Set(['bicycle', 'car', 'motorcycle', 'bus', 'truck', 'bench',
  'chair', 'potted plant', 'fire hydrant', 'parking meter', 'suitcase', 'couch', 'dining table']);
let modelPromise;
let queue = Promise.resolve();

async function load() {
  await tf.setBackend('cpu');
  await tf.ready();
  return cocoSsd.load({base: 'lite_mobilenet_v2',
    modelUrl: new URL('detector-assets/model.json', self.location.href).href});
}

async function infer({width, height, rgba}) {
  modelPromise ??= load().catch(error => { modelPromise = undefined; throw error; });
  const model = await modelPromise;
  const bytes = new Uint8Array(rgba);
  const rgb = new Uint8Array(width * height * 3);
  for (let src = 0, dst = 0; src < bytes.length; src += 4, dst += 3) {
    rgb[dst] = bytes[src];
    rgb[dst+1] = bytes[src+1];
    rgb[dst+2] = bytes[src+2];
  }
  const input = tf.tensor3d(rgb, [height, width, 3], 'int32');
  try {
    const found = await model.detect(input, 40, 0.4);
    const clamp = n => Math.max(0, Math.min(1, n));
    return found.filter(obj => fieldObjects.has(obj.class)).map(obj => {
      const [x, y, w, h] = obj.bbox;
      return {label: obj.class, score: obj.score,
        box: [clamp(x/width), clamp(y/height), clamp((x+w)/width), clamp((y+h)/height)]};
    }).filter(obj => obj.box[2] > obj.box[0] && obj.box[3] > obj.box[1]);
  } finally {
    input.dispose();
  }
}

self.onmessage = ({data}) => {
  queue = queue.then(async () => {
    try { self.postMessage({id: data.id, items: await infer(data)}); }
    catch (error) { self.postMessage({id: data.id, error: String(error.message || error)}); }
  });
};
