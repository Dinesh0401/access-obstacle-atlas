// Run inference off the UI thread. The source image and all pixel data stay local.
let worker;
let sequence = 0;
const pending = new Map();

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./detector-worker.js', import.meta.url));
  worker.onmessage = ({data}) => {
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.items);
  };
  worker.onerror = event => {
    for (const {reject} of pending.values()) reject(new Error(event.message || 'Detector worker failed'));
    pending.clear();
    worker.terminate();
    worker = undefined;
  };
  return worker;
}

export async function detect(image) {
  if (!image.width || !image.height) throw new Error('The image has no pixels.');
  const ctx = image.getContext('2d', {willReadFrequently: true});
  const frame = ctx.getImageData(0, 0, image.width, image.height);
  const target = getWorker();
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    target.postMessage({id, width: frame.width, height: frame.height, rgba: frame.data.buffer},
      [frame.data.buffer]);
  });
}
