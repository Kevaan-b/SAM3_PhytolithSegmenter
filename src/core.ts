import type { PointPrompt } from "./protocol";

export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function fitDimensions(
  width: number,
  height: number,
  maxWidth: number,
  maxHeight: number,
): { width: number; height: number } {
  if (
    width <= 0 ||
    height <= 0 ||
    maxWidth <= 0 ||
    maxHeight <= 0
  ) {
    throw new Error("Image and container dimensions must be positive.");
  }

  const scale = Math.min(maxWidth / width, maxHeight / height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export function normalizePointer(
  clientX: number,
  clientY: number,
  rect: RectLike,
): Pick<PointPrompt, "x" | "y"> {
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error("Cannot normalize a point against an empty image.");
  }

  return {
    x: clamp((clientX - rect.left) / rect.width, 0, 1),
    y: clamp((clientY - rect.top) / rect.height, 0, 1),
  };
}

export function toModelPoint(
  point: PointPrompt,
  resizedHeight: number,
  resizedWidth: number,
): [number, number] {
  return [point.x * resizedWidth, point.y * resizedHeight];
}

export function makePromptData(
  points: readonly PointPrompt[],
  resizedHeight: number,
  resizedWidth: number,
): {
  coordinates: Float32Array;
  labels: BigInt64Array;
  pointShape: [number, number, number, number];
  labelShape: [number, number, number];
} {
  const coordinates = new Float32Array(points.length * 2);
  const labels = new BigInt64Array(points.length);

  points.forEach((point, index) => {
    const [x, y] = toModelPoint(point, resizedHeight, resizedWidth);
    coordinates[index * 2] = x;
    coordinates[index * 2 + 1] = y;
    labels[index] = BigInt(point.label);
  });

  return {
    coordinates,
    labels,
    pointShape: [1, 1, points.length, 2],
    labelShape: [1, 1, points.length],
  };
}

export function argmax(values: ArrayLike<number>): number {
  if (values.length === 0) {
    throw new Error("Cannot choose a mask from an empty score array.");
  }

  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if ((values[index] ?? -Infinity) > (values[best] ?? -Infinity)) {
      best = index;
    }
  }
  return best;
}

export function promptKey(
  points: readonly PointPrompt[],
  resizedSize = 1008,
): string {
  return points
    .map(
      ({ x, y, label }) =>
        `${Math.round(x * resizedSize)}:${Math.round(y * resizedSize)}:${label}`,
    )
    .join("|");
}

export function removeLastPoint(
  points: readonly PointPrompt[],
): PointPrompt[] {
  return points.slice(0, -1);
}

export function clearPoints(): PointPrompt[] {
  return [];
}

export function composePrompts(
  pinned: readonly PointPrompt[],
  hover: PointPrompt | null,
): PointPrompt[] {
  return hover ? [...pinned, hover] : [...pinned];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
