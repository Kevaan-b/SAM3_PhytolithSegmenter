export interface CompositeLayer {
  mask: Uint8Array;
  color: readonly [number, number, number];
  alpha: number;
}

export function outerMaskOutline(
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Uint8Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error("Mask dimensions must be positive integers.");
  }
  const pixelCount = width * height;
  if (mask.byteLength !== Math.ceil(pixelCount / 8)) throw new Error("Mask dimensions do not match.");
  const distance = Math.max(1, Math.floor(radius));
  const stride = width + 1;
  const integral = new Uint32Array(stride * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowTotal = 0;
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      rowTotal += (mask[index >> 3]! >> (index & 7)) & 1;
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1]! + rowTotal;
    }
  }
  const outline = new Uint8Array(mask.byteLength);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - distance);
    const bottom = Math.min(height, y + distance + 1);
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if ((mask[index >> 3]! & (1 << (index & 7))) !== 0) continue;
      const left = Math.max(0, x - distance);
      const right = Math.min(width, x + distance + 1);
      const nearby = integral[bottom * stride + right]!
        - integral[top * stride + right]!
        - integral[bottom * stride + left]!
        + integral[top * stride + left]!;
      if (nearby > 0) outline[index >> 3] = outline[index >> 3]! | (1 << (index & 7));
    }
  }
  return outline;
}

export function excludeOverlaps(
  mask: Uint8Array,
  blockers: readonly Uint8Array[],
  pixelCount: number,
): Uint8Array {
  const byteLength = Math.ceil(pixelCount / 8);
  if (mask.byteLength !== byteLength || blockers.some((item) => item.byteLength !== byteLength)) {
    throw new Error("Mask dimensions do not match.");
  }
  const output = mask.slice();
  for (const blocker of blockers) {
    for (let index = 0; index < byteLength; index += 1) {
      output[index] = output[index]! & ~blocker[index]!;
    }
  }
  const remainder = pixelCount & 7;
  if (remainder !== 0) output[byteLength - 1] = output[byteLength - 1]! & ((1 << remainder) - 1);
  return output;
}

export function compositeMasks(
  target: Uint8ClampedArray,
  pixelCount: number,
  layers: readonly CompositeLayer[],
): void {
  if (target.length < pixelCount * 4) throw new Error("RGBA target is too small.");
  target.fill(0);
  for (const layer of layers) compositeLayer(target, pixelCount, layer);
}

function compositeLayer(
  target: Uint8ClampedArray,
  pixelCount: number,
  layer: CompositeLayer,
): void {
  const sourceAlpha = layer.alpha;
  const sourceAlphaByte = Math.round(sourceAlpha * 255);
  const outputAlpha = new Uint8Array(256);
  const sourceWeight = new Float32Array(256);
  for (let destinationAlphaByte = 0; destinationAlphaByte < 256; destinationAlphaByte += 1) {
    const destinationAlpha = destinationAlphaByte / 255;
    const output = sourceAlpha + destinationAlpha * (1 - sourceAlpha);
    outputAlpha[destinationAlphaByte] = Math.round(output * 255);
    sourceWeight[destinationAlphaByte] = sourceAlpha / output;
  }
  for (let index = 0; index < pixelCount; index += 1) {
    if ((layer.mask[index >> 3]! & (1 << (index & 7))) === 0) continue;
    const offset = index * 4;
    const destinationAlphaByte = target[offset + 3]!;
    if (destinationAlphaByte === 0) {
      target[offset] = layer.color[0];
      target[offset + 1] = layer.color[1];
      target[offset + 2] = layer.color[2];
      target[offset + 3] = sourceAlphaByte;
      continue;
    }
    const weight = sourceWeight[destinationAlphaByte]!;
    const priorWeight = 1 - weight;
    target[offset] = Math.round(layer.color[0] * weight + target[offset]! * priorWeight);
    target[offset + 1] = Math.round(layer.color[1] * weight + target[offset + 1]! * priorWeight);
    target[offset + 2] = Math.round(layer.color[2] * weight + target[offset + 2]! * priorWeight);
    target[offset + 3] = outputAlpha[destinationAlphaByte]!;
  }
}
