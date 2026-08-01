export interface CompositeLayer {
  mask: Uint8Array;
  color: readonly [number, number, number];
  alpha: number;
}

const alphaTables = new Map<number, {
  sourceAlphaByte: number;
  outputAlpha: Uint8Array;
  sourceWeight: Float32Array;
}>();

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
  const horizontal = new Uint8Array(pixelCount);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    let nearby = 0;
    for (let x = 0; x < Math.min(width, distance + 1); x += 1) {
      nearby += (mask[(row + x) >> 3]! >> ((row + x) & 7)) & 1;
    }
    for (let x = 0; x < width; x += 1) {
      horizontal[row + x] = nearby > 0 ? 1 : 0;
      const remove = x - distance;
      if (remove >= 0) nearby -= (mask[(row + remove) >> 3]! >> ((row + remove) & 7)) & 1;
      const add = x + distance + 1;
      if (add < width) nearby += (mask[(row + add) >> 3]! >> ((row + add) & 7)) & 1;
    }
  }

  const verticalCounts = new Uint32Array(width);
  for (let y = 0; y < Math.min(height, distance + 1); y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      verticalCounts[x] = verticalCounts[x]! + horizontal[row + x]!;
    }
  }
  const outline = new Uint8Array(mask.byteLength);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const index = row + x;
      if ((mask[index >> 3]! & (1 << (index & 7))) !== 0) continue;
      if (verticalCounts[x]! > 0) outline[index >> 3] = outline[index >> 3]! | (1 << (index & 7));
    }
    const remove = y - distance;
    if (remove >= 0) {
      const removeRow = remove * width;
      for (let x = 0; x < width; x += 1) {
        verticalCounts[x] = verticalCounts[x]! - horizontal[removeRow + x]!;
      }
    }
    const add = y + distance + 1;
    if (add < height) {
      const addRow = add * width;
      for (let x = 0; x < width; x += 1) {
        verticalCounts[x] = verticalCounts[x]! + horizontal[addRow + x]!;
      }
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
  const table = alphaTable(layer.alpha);
  for (let byteIndex = 0; byteIndex < layer.mask.length; byteIndex += 1) {
    let bits = layer.mask[byteIndex]!;
    if (bits === 0) continue;
    const firstPixel = byteIndex * 8;
    for (let bit = 0; bits !== 0 && bit < 8; bit += 1, bits >>= 1) {
      if ((bits & 1) === 0) continue;
      const pixelIndex = firstPixel + bit;
      if (pixelIndex >= pixelCount) break;
      const offset = pixelIndex * 4;
      const destinationAlphaByte = target[offset + 3]!;
      if (destinationAlphaByte === 0) {
        target[offset] = layer.color[0];
        target[offset + 1] = layer.color[1];
        target[offset + 2] = layer.color[2];
        target[offset + 3] = table.sourceAlphaByte;
        continue;
      }
      const weight = table.sourceWeight[destinationAlphaByte]!;
      const priorWeight = 1 - weight;
      target[offset] = Math.round(layer.color[0] * weight + target[offset]! * priorWeight);
      target[offset + 1] = Math.round(layer.color[1] * weight + target[offset + 1]! * priorWeight);
      target[offset + 2] = Math.round(layer.color[2] * weight + target[offset + 2]! * priorWeight);
      target[offset + 3] = table.outputAlpha[destinationAlphaByte]!;
    }
  }
}

function alphaTable(sourceAlpha: number): {
  sourceAlphaByte: number;
  outputAlpha: Uint8Array;
  sourceWeight: Float32Array;
} {
  const cached = alphaTables.get(sourceAlpha);
  if (cached) return cached;
  const table = {
    sourceAlphaByte: Math.round(sourceAlpha * 255),
    outputAlpha: new Uint8Array(256),
    sourceWeight: new Float32Array(256),
  };
  for (let destinationAlphaByte = 0; destinationAlphaByte < 256; destinationAlphaByte += 1) {
    const destinationAlpha = destinationAlphaByte / 255;
    const output = sourceAlpha + destinationAlpha * (1 - sourceAlpha);
    table.outputAlpha[destinationAlphaByte] = Math.round(output * 255);
    table.sourceWeight[destinationAlphaByte] = sourceAlpha / output;
  }
  alphaTables.set(sourceAlpha, table);
  return table;
}
