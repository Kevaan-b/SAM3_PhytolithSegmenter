import { describe, expect, it } from "vitest";
import { MaskEditor } from "./mask-editor";

function packed(width: number, height: number, enabled: number[]): Uint8Array {
  const mask = new Uint8Array(Math.ceil((width * height) / 8));
  for (const index of enabled) {
    const byte = index >> 3;
    mask[byte] = (mask[byte] ?? 0) | (1 << (index & 7));
  }
  return mask;
}

function enabled(mask: Uint8Array, count: number): number[] {
  return Array.from({ length: count }, (_, index) => index).filter(
    (index) => ((mask[index >> 3] ?? 0) & (1 << (index & 7))) !== 0,
  );
}

describe("MaskEditor", () => {
  it("adds and erases clipped circular strokes", () => {
    const editor = new MaskEditor();
    editor.resetImage(8, 8);
    editor.setBaseMask(packed(8, 8, [27, 28, 35, 36]));

    editor.beginStroke("add", 1.1, { x: 0, y: 0 });
    editor.endStroke();
    expect(enabled(editor.displayedMask(), 64)).toEqual(
      expect.arrayContaining([0, 1, 8, 27, 28, 35, 36]),
    );

    editor.beginStroke("erase", 1, { x: 3 / 7, y: 3 / 7 });
    editor.endStroke();
    expect(enabled(editor.displayedMask(), 64)).not.toContain(27);
  });

  it("interpolates continuous strokes", () => {
    const editor = new MaskEditor();
    editor.resetImage(16, 3);
    editor.setBaseMask(packed(16, 3, [16]));
    editor.beginStroke("add", 1, { x: 0, y: 0.5 });
    editor.endStroke([{ x: 1, y: 0.5 }]);
    const result = enabled(editor.displayedMask(), 48);
    for (let x = 0; x < 16; x += 1) expect(result).toContain(16 + x);
  });

  it("preserves overrides across new SAM masks", () => {
    const editor = new MaskEditor();
    editor.resetImage(4, 4);
    editor.setBaseMask(packed(4, 4, [0]));
    editor.beginStroke("add", 0.6, { x: 1, y: 1 });
    editor.endStroke();
    editor.setBaseMask(packed(4, 4, [5]));
    expect(enabled(editor.displayedMask(), 16)).toEqual([5, 15]);
  });

  it("inverts exactly and keeps brushes intuitive while inverted", () => {
    const editor = new MaskEditor();
    editor.resetImage(3, 3);
    editor.setBaseMask(packed(3, 3, [4]));
    editor.toggleInvert();
    const inverted = editor.displayedMask();
    expect(enabled(inverted, 9)).toEqual([0, 1, 2, 3, 5, 6, 7, 8]);
    expect(inverted[1]).toBe(1); // Seven padding bits remain clear.

    editor.beginStroke("erase", 0.6, { x: 0, y: 0 });
    editor.endStroke();
    editor.beginStroke("add", 0.6, { x: 0.5, y: 0.5 });
    editor.endStroke();
    expect(enabled(editor.displayedMask(), 9)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("undoes whole strokes and inversion, then resets edits", () => {
    const editor = new MaskEditor();
    editor.resetImage(4, 4);
    editor.setBaseMask(packed(4, 4, [0]));
    editor.beginStroke("add", 0.6, { x: 1, y: 1 });
    editor.extendStroke([{ x: 2 / 3, y: 1 }]);
    editor.endStroke();
    editor.toggleInvert();

    expect(editor.state()).toMatchObject({ canUndo: true, inverted: true });
    expect(editor.undo()).toBe(true);
    expect(editor.state().inverted).toBe(false);
    expect(editor.undo()).toBe(true);
    expect(enabled(editor.displayedMask(), 16)).toEqual([0]);

    editor.toggleInvert();
    editor.resetEdits();
    expect(editor.state()).toEqual({
      hasMask: true,
      hasEdits: false,
      canUndo: false,
      inverted: false,
    });
  });

  it("caps history and clears all state with the image mask", () => {
    const editor = new MaskEditor(2);
    editor.resetImage(4, 4);
    editor.setBaseMask(packed(4, 4, [0]));
    editor.toggleInvert();
    editor.toggleInvert();
    editor.toggleInvert();
    expect(editor.undo()).toBe(true);
    expect(editor.undo()).toBe(true);
    expect(editor.undo()).toBe(false);
    editor.clearMask();
    expect(editor.state()).toEqual({
      hasMask: false,
      hasEdits: false,
      canUndo: false,
      inverted: false,
    });
  });
});
