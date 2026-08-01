export type BrushOperation = "add" | "erase";

export interface MaskPoint {
  x: number;
  y: number;
}

interface Snapshot {
  forceOn: Uint8Array;
  forceOff: Uint8Array;
  inverted: boolean;
  serial: number;
}

export interface MaskEditState {
  hasMask: boolean;
  hasEdits: boolean;
  canUndo: boolean;
  inverted: boolean;
}

export class MaskEditor {
  private static nextHistorySerial = 0;
  private readonly historyLimit: number;
  private width = 0;
  private height = 0;
  private base = new Uint8Array();
  private forceOn = new Uint8Array();
  private forceOff = new Uint8Array();
  private inverted = false;
  private history: Snapshot[] = [];
  private strokeSnapshot: Snapshot | undefined;
  private strokeOperation: BrushOperation | undefined;
  private strokeRadius = 0;
  private lastPoint: MaskPoint | undefined;
  private strokeChanged = false;

  constructor(historyLimit = 100) {
    if (historyLimit < 1) {
      throw new Error("Mask edit history limit must be positive.");
    }
    this.historyLimit = historyLimit;
  }

  resetImage(width: number, height: number): void {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
      throw new Error("Mask dimensions must be positive integers.");
    }
    this.width = width;
    this.height = height;
    const byteLength = Math.ceil((width * height) / 8);
    this.base = new Uint8Array(byteLength);
    this.forceOn = new Uint8Array(byteLength);
    this.forceOff = new Uint8Array(byteLength);
    this.inverted = false;
    this.history = [];
    this.cancelStroke();
  }

  setBaseMask(mask: Uint8Array): void {
    this.requireDimensions();
    if (mask.byteLength !== this.base.byteLength) {
      throw new Error("Base mask size does not match the current image.");
    }
    this.base.set(mask);
    this.clearPaddingBits(this.base);
  }

  clearBaseMask(): void {
    if (this.width > 0) this.base.fill(0);
  }

  baseMask(): Uint8Array {
    return this.base.slice();
  }

  clearMask(): void {
    if (this.width === 0) return;
    this.base.fill(0);
    this.forceOn.fill(0);
    this.forceOff.fill(0);
    this.inverted = false;
    this.history = [];
    this.cancelStroke();
  }

  beginStroke(
    operation: BrushOperation,
    radius: number,
    point: MaskPoint,
  ): void {
    this.requireMask();
    if (!Number.isFinite(radius) || radius <= 0) {
      throw new Error("Brush radius must be positive.");
    }
    this.cancelStroke();
    this.strokeSnapshot = this.snapshot();
    this.strokeOperation = operation;
    this.strokeRadius = radius;
    this.lastPoint = point;
    this.strokeChanged = this.stamp(point, operation, radius);
  }

  extendStroke(points: readonly MaskPoint[]): void {
    if (!this.strokeSnapshot || !this.strokeOperation || !this.lastPoint) return;
    for (const point of points) {
      this.strokeChanged =
        this.segment(
          this.lastPoint,
          point,
          this.strokeOperation,
          this.strokeRadius,
        ) || this.strokeChanged;
      this.lastPoint = point;
    }
  }

  endStroke(points: readonly MaskPoint[] = []): boolean {
    this.extendStroke(points);
    if (this.strokeSnapshot && this.strokeChanged) {
      this.pushHistory(this.strokeSnapshot);
    }
    const changed = this.strokeChanged;
    this.cancelStroke();
    return changed;
  }

  toggleInvert(): void {
    this.requireMask();
    this.pushHistory(this.snapshot());
    this.inverted = !this.inverted;
  }

  undo(): boolean {
    const snapshot = this.history.pop();
    if (!snapshot) return false;
    this.forceOn.set(snapshot.forceOn);
    this.forceOff.set(snapshot.forceOff);
    this.inverted = snapshot.inverted;
    this.cancelStroke();
    return true;
  }

  resetEdits(): void {
    this.forceOn.fill(0);
    this.forceOff.fill(0);
    this.inverted = false;
    this.history = [];
    this.cancelStroke();
  }

  historyBytes(): number {
    return this.history.reduce(
      (total, snapshot) =>
        total + snapshot.forceOn.byteLength + snapshot.forceOff.byteLength + 9,
      0,
    );
  }

  oldestHistorySerial(): number | undefined {
    return this.history[0]?.serial;
  }

  discardOldestHistory(): boolean {
    return this.history.shift() !== undefined;
  }

  state(): MaskEditState {
    return {
      hasMask: this.hasMask(),
      hasEdits:
        this.inverted ||
        this.forceOn.some((value) => value !== 0) ||
        this.forceOff.some((value) => value !== 0),
      canUndo: this.history.length > 0,
      inverted: this.inverted,
    };
  }

  writeAlpha(target: Uint8ClampedArray, alpha: number): void {
    this.requireDimensions();
    if (target.length < this.width * this.height * 4) {
      throw new Error("RGBA target is smaller than the current mask.");
    }
    const count = this.width * this.height;
    for (let index = 0; index < count; index += 1) {
      target[index * 4 + 3] = this.displayedAt(index) ? alpha : 0;
    }
  }

  displayedMask(base: Uint8Array = this.base): Uint8Array {
    this.requireDimensions();
    if (base.byteLength !== this.base.byteLength) {
      throw new Error("Base mask size does not match the current image.");
    }
    const output = new Uint8Array(this.base.length);
    for (let index = 0; index < this.width * this.height; index += 1) {
      if (this.displayedAt(index, base)) setBit(output, index, true);
    }
    return output;
  }

  private hasMask(): boolean {
    return this.base.some((value) => value !== 0);
  }

  private requireMask(): void {
    this.requireDimensions();
    if (!this.hasMask()) throw new Error("A generated mask is required before editing.");
  }

  private requireDimensions(): void {
    if (this.width === 0 || this.height === 0) {
      throw new Error("Mask image has not been initialized.");
    }
  }

  private displayedAt(index: number, base: Uint8Array = this.base): boolean {
    const raw =
      (getBit(base, index) || getBit(this.forceOn, index)) &&
      !getBit(this.forceOff, index);
    return this.inverted ? !raw : raw;
  }

  private stamp(
    point: MaskPoint,
    operation: BrushOperation,
    radius: number,
  ): boolean {
    const centerX = clamp(point.x, 0, 1) * (this.width - 1);
    const centerY = clamp(point.y, 0, 1) * (this.height - 1);
    const minimumX = Math.max(0, Math.floor(centerX - radius));
    const maximumX = Math.min(this.width - 1, Math.ceil(centerX + radius));
    const minimumY = Math.max(0, Math.floor(centerY - radius));
    const maximumY = Math.min(this.height - 1, Math.ceil(centerY + radius));
    const radiusSquared = radius * radius;
    const displayedValue = operation === "add";
    const rawValue = this.inverted ? !displayedValue : displayedValue;
    let changed = false;

    for (let y = minimumY; y <= maximumY; y += 1) {
      for (let x = minimumX; x <= maximumX; x += 1) {
        const dx = x - centerX;
        const dy = y - centerY;
        if (dx * dx + dy * dy > radiusSquared) continue;
        const index = y * this.width + x;
        const previousOn = getBit(this.forceOn, index);
        const previousOff = getBit(this.forceOff, index);
        setBit(this.forceOn, index, rawValue);
        setBit(this.forceOff, index, !rawValue);
        changed =
          changed ||
          previousOn !== rawValue ||
          previousOff !== !rawValue;
      }
    }
    return changed;
  }

  private segment(
    from: MaskPoint,
    to: MaskPoint,
    operation: BrushOperation,
    radius: number,
  ): boolean {
    const dx = (to.x - from.x) * (this.width - 1);
    const dy = (to.y - from.y) * (this.height - 1);
    const distance = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius / 2)));
    let changed = false;
    for (let step = 1; step <= steps; step += 1) {
      const progress = step / steps;
      changed =
        this.stamp(
          {
            x: from.x + (to.x - from.x) * progress,
            y: from.y + (to.y - from.y) * progress,
          },
          operation,
          radius,
        ) || changed;
    }
    return changed;
  }

  private snapshot(): Snapshot {
    return {
      forceOn: this.forceOn.slice(),
      forceOff: this.forceOff.slice(),
      inverted: this.inverted,
      serial: ++MaskEditor.nextHistorySerial,
    };
  }

  private pushHistory(snapshot: Snapshot): void {
    this.history.push(snapshot);
    if (this.history.length > this.historyLimit) this.history.shift();
  }

  private cancelStroke(): void {
    this.strokeSnapshot = undefined;
    this.strokeOperation = undefined;
    this.strokeRadius = 0;
    this.lastPoint = undefined;
    this.strokeChanged = false;
  }

  private clearPaddingBits(mask: Uint8Array): void {
    const remainder = (this.width * this.height) & 7;
    if (remainder === 0 || mask.length === 0) return;
    const last = mask.length - 1;
    mask[last] = (mask[last] ?? 0) & ((1 << remainder) - 1);
  }
}

function getBit(mask: Uint8Array, index: number): boolean {
  return ((mask[index >> 3] ?? 0) & (1 << (index & 7))) !== 0;
}

function setBit(mask: Uint8Array, index: number, value: boolean): void {
  const byte = index >> 3;
  const bit = 1 << (index & 7);
  if (value) {
    mask[byte] = (mask[byte] ?? 0) | bit;
  } else {
    mask[byte] = (mask[byte] ?? 0) & ~bit;
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
