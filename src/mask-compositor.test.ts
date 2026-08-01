import { describe, expect, it } from "vitest";
import { compositeMasks, excludeOverlaps, outerMaskOutline } from "./mask-compositor";

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

  it("keeps committed-only pixels visible while a normal hover mask is composited", () => {
    const pixels = new Uint8ClampedArray(12);
    const committed = Uint8Array.of(0b011);
    const preview = Uint8Array.of(0b110);
    const committedOnly = excludeOverlaps(committed, [preview], 3);
    compositeMasks(pixels, 3, [
      { mask: committedOnly, color: [64, 148, 220], alpha: 0.34 },
      { mask: preview, color: [64, 148, 220], alpha: 0.5 },
    ]);

    expect([...pixels.slice(0, 4)]).toEqual([64, 148, 220, 87]);
    expect([...pixels.slice(4, 8)]).toEqual([64, 148, 220, 128]);
    expect([...pixels.slice(8, 12)]).toEqual([64, 148, 220, 128]);
  });

});

describe("outerMaskOutline", () => {
  it("creates a clipped outer boundary without covering the mask", () => {
    const centered = outerMaskOutline(Uint8Array.of(1 << 4, 0), 3, 3, 1);
    expect(centered[0]).toBe(0b11101111);
    expect(centered[1]).toBe(0b1);

    const corner = outerMaskOutline(Uint8Array.of(1, 0), 3, 3, 1);
    expect(corner[0]).toBe(0b00011010);
    expect(corner[1]).toBe(0);
  });

  it("rejects mismatched dimensions", () => {
    expect(() => outerMaskOutline(new Uint8Array(1), 3, 3, 1)).toThrow(/dimensions/i);
  });

  it("matches a direct neighborhood scan for wider outlines", () => {
    const width = 7;
    const height = 5;
    const source = Uint8Array.of(0b01001001, 0b10010010, 0b00100100, 0b01001001, 0);
    const actual = outerMaskOutline(source, width, height, 2);
    const expected = new Uint8Array(source.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        if ((source[index >> 3]! & (1 << (index & 7))) !== 0) continue;
        let nearby = false;
        for (let checkY = Math.max(0, y - 2); checkY <= Math.min(height - 1, y + 2); checkY += 1) {
          for (let checkX = Math.max(0, x - 2); checkX <= Math.min(width - 1, x + 2); checkX += 1) {
            const check = checkY * width + checkX;
            nearby ||= (source[check >> 3]! & (1 << (check & 7))) !== 0;
          }
        }
        if (nearby) expected[index >> 3] = expected[index >> 3]! | (1 << (index & 7));
      }
    }
    expect(actual).toEqual(expected);
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
