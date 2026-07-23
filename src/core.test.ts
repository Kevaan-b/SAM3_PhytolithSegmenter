import { describe, expect, it } from "vitest";
import {
  argmax,
  clearPoints,
  composePrompts,
  fitDimensions,
  makePromptData,
  normalizePointer,
  promptKey,
  removeLastPoint,
  toModelPoint,
} from "./core";
import type { PointPrompt } from "./protocol";

describe("point coordinate helpers", () => {
  it("fits an image without changing its aspect ratio", () => {
    expect(fitDimensions(1264, 1268, 1600, 900)).toEqual({
      width: 897,
      height: 900,
    });
    expect(fitDimensions(1600, 800, 1000, 1000)).toEqual({
      width: 1000,
      height: 500,
    });
  });

  it("normalizes screen coordinates against the displayed image bounds", () => {
    expect(
      normalizePointer(350, 225, {
        left: 100,
        top: 100,
        width: 500,
        height: 250,
      }),
    ).toEqual({ x: 0.5, y: 0.5 });
  });

  it("clamps points to the image boundary", () => {
    expect(
      normalizePointer(-10, 900, {
        left: 10,
        top: 20,
        width: 100,
        height: 200,
      }),
    ).toEqual({ x: 0, y: 1 });
  });

  it("rejects empty image bounds", () => {
    expect(() =>
      normalizePointer(0, 0, {
        left: 0,
        top: 0,
        width: 0,
        height: 100,
      }),
    ).toThrow(/empty image/i);
  });

  it("scales normalized coordinates in width, height order", () => {
    expect(
      toModelPoint({ x: 0.25, y: 0.75, label: 1 }, 800, 1000),
    ).toEqual([250, 600]);
  });
});

describe("prompt construction", () => {
  const points: PointPrompt[] = [
    { x: 0.2, y: 0.3, label: 1 },
    { x: 0.8, y: 0.7, label: 0 },
  ];

  it("creates the SAM point and label tensor data", () => {
    const prompt = makePromptData(points, 800, 1000);

    expect([...prompt.coordinates]).toEqual([200, 240, 800, 560]);
    expect([...prompt.labels]).toEqual([1n, 0n]);
    expect(prompt.pointShape).toEqual([1, 1, 2, 2]);
    expect(prompt.labelShape).toEqual([1, 1, 2]);
  });

  it("deduplicates pointer prompts at model-pixel precision", () => {
    const a = promptKey([{ x: 0.5001, y: 0.5001, label: 1 }]);
    const b = promptKey([{ x: 0.50011, y: 0.50011, label: 1 }]);
    const negative = promptKey([{ x: 0.5001, y: 0.5001, label: 0 }]);

    expect(a).toBe(b);
    expect(a).not.toBe(negative);
  });
});

describe("mask and point state helpers", () => {
  it("selects the candidate with the highest score", () => {
    expect(argmax(new Float32Array([0.12, 0.91, 0.45]))).toBe(1);
  });

  it("supports undo and clear without mutating the original list", () => {
    const points: PointPrompt[] = [
      { x: 0.2, y: 0.3, label: 1 },
      { x: 0.8, y: 0.7, label: 0 },
    ];

    expect(removeLastPoint(points)).toEqual([points[0]]);
    expect(clearPoints()).toEqual([]);
    expect(points).toHaveLength(2);
  });

  it("uses pinned points alone after the pointer leaves", () => {
    const pinned: PointPrompt[] = [{ x: 0.2, y: 0.3, label: 1 }];
    const hover: PointPrompt = { x: 0.8, y: 0.7, label: 0 };

    expect(composePrompts(pinned, hover)).toEqual([pinned[0], hover]);
    expect(composePrompts(pinned, null)).toEqual(pinned);
    expect(composePrompts(pinned, null)).not.toBe(pinned);
  });
});
