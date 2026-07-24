import { describe, expect, it } from "vitest";
import { MASK_PALETTE, MaskLayerCollection } from "./mask-layers";

function collection(): MaskLayerCollection {
  let id = 0;
  return new MaskLayerCollection(() => `layer-${++id}`);
}

describe("MaskLayerCollection", () => {
  it("always starts with one visible active layer", () => {
    const layers = collection();
    expect(layers.all()).toHaveLength(1);
    expect(layers.active()).toMatchObject({ id: "layer-1", name: "object", categoryId: 1, color: MASK_PALETTE[0], visible: true });
  });

  it("adds unlimited independently named and colored layers", () => {
    const layers = collection();
    for (let index = 0; index < MASK_PALETTE.length + 2; index += 1) layers.add();
    expect(layers.all()).toHaveLength(MASK_PALETTE.length + 3);
    expect(layers.active().name).toBe(`Mask ${MASK_PALETTE.length + 3}`);
    expect(layers.all()[MASK_PALETTE.length]!.color).toBe(MASK_PALETTE[0]);
    layers.rename(layers.active().id, " Class A ");
    layers.rename(layers.all()[0]!.id, "Class A");
    expect(layers.active().name).toBe("Class A");
  });

  it("keeps selection and visibility independent", () => {
    const layers = collection();
    const first = layers.active();
    const second = layers.add();
    layers.setVisible(second.id, false);
    expect(layers.active()).toMatchObject({ id: second.id, visible: false });
    layers.select(first.id);
    expect(layers.active().visible).toBe(true);
    expect(layers.all()[1]!.visible).toBe(false);
  });

  it("deletes a layer and selects the nearest survivor", () => {
    const layers = collection();
    const second = layers.add();
    const third = layers.add();
    layers.select(second.id);
    expect(layers.delete(second.id).id).toBe(third.id);
    layers.delete(third.id);
    expect(() => layers.delete(layers.active().id)).toThrow(/at least one/i);
  });

  it("validates names and colors", () => {
    const layers = collection();
    expect(() => layers.rename(layers.active().id, "   ")).toThrow(/empty/i);
    expect(() => layers.setColor(layers.active().id, "blue")).toThrow(/RRGGBB/);
    layers.setColor(layers.active().id, "#AABBCC");
    expect(layers.active().color).toBe("#aabbcc");
  });
});
