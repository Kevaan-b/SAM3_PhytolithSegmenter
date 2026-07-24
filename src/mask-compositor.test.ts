import { describe, expect, it } from "vitest";
import { compositeMasks, excludeOverlaps } from "./mask-compositor";

describe("compositeMasks", () => {
  it("skips hidden layers supplied by the caller and blends overlaps in order", () => {
    const pixels = new Uint8ClampedArray(8);
    compositeMasks(pixels, 2, [
      { mask: Uint8Array.of(0b11), color: [255, 0, 0], alpha: 0.34 },
      { mask: Uint8Array.of(0b01), color: [0, 0, 255], alpha: 0.5 },
    ]);
    expect([...pixels.slice(4, 8)]).toEqual([255, 0, 0, 87]);
    expect(pixels[2]).toBeGreaterThan(pixels[0]!);
    expect(pixels[3]).toBeGreaterThan(127);
  });

  it("clears the prior frame when no layers are visible", () => {
    const pixels = new Uint8ClampedArray([1, 2, 3, 4]);
    compositeMasks(pixels, 1, []);
    expect([...pixels]).toEqual([0, 0, 0, 0]);
  });
});

describe("excludeOverlaps", () => {
  it("removes every blocker from the latest mask without mutating its source", () => {
    const source = Uint8Array.of(0b00000011, 0b00000011);
    const result = excludeOverlaps(
      source,
      [Uint8Array.of(0b00000010, 0), Uint8Array.of(0, 0b00000011)],
      10,
    );
    expect([...result]).toEqual([0b00000001, 0]);
    expect([...source]).toEqual([0b00000011, 0b00000011]);
  });

  it("rejects mismatched mask dimensions", () => {
    expect(() => excludeOverlaps(Uint8Array.of(1), [new Uint8Array(2)], 8)).toThrow(/dimensions/i);
  });
});
