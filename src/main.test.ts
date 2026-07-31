// @vitest-environment jsdom

import { beforeAll, describe, expect, it, vi } from "vitest";
import type {
  MainToWorkerMessage,
  WorkerToMainMessage,
} from "./protocol";
import type { DataFolder } from "./data-navigator";

const DATA_TREE: DataFolder = {
  name: "Data",
  path: "",
  images: [],
  folders: [
    {
      name: "train",
      path: "train",
      folders: [
        {
          name: "nested",
          path: "train/nested",
          folders: [],
          images: [],
        },
      ],
      images: [
        {
          id: "train-a",
          name: "train-a.png",
          path: "train/train-a.png",
          url: "/data/train/train-a.png",
          cacheState: "ready",
        },
        {
          id: "train-b",
          name: "train-b.png",
          path: "train/train-b.png",
          url: "/data/train/train-b.png",
          cacheState: "ready",
        },
      ],
    },
    {
      name: "val",
      path: "val",
      folders: [],
      images: [
        {
          id: "val-a",
          name: "val-a.png",
          path: "val/val-a.png",
          url: "/data/val/val-a.png",
          cacheState: "ready",
        },
      ],
    },
  ],
};

class MockWorker {
  static instance: MockWorker;

  onmessage: ((event: MessageEvent<WorkerToMainMessage>) => void) | null =
    null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly messages: MainToWorkerMessage[] = [];

  constructor() {
    MockWorker.instance = this;
  }

  postMessage(message: MainToWorkerMessage): void {
    this.messages.push(message);
    if (message.type === "snapshot-annotations") {
      const load = [...this.messages].reverse().find(
        (item): item is Extract<MainToWorkerMessage, { type: "load-image" }> => item.type === "load-image",
      );
      queueMicrotask(() => this.emit({
        type: "annotation-snapshot",
        requestId: message.requestId,
        imageRevision: message.imageRevision,
        width: 982,
        height: 996,
        latestMaskLayerId: load?.activeLayerId ?? "",
        layers: (load?.layers ?? []).map((layer) => ({
          layerId: layer.id,
          rawMask: new Uint8Array(Math.ceil((982 * 996) / 8)),
          effectiveMask: new Uint8Array(Math.ceil((982 * 996) / 8)),
        })),
      }));
    }
  }

  emit(message: WorkerToMainMessage): void {
    this.onmessage?.(
      new MessageEvent<WorkerToMainMessage>("message", { data: message }),
    );
  }
}

describe("browser UI with a mocked inference worker", () => {
  beforeAll(async () => {
    document.body.innerHTML = '<div id="app"></div>';
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(
      HTMLCanvasElement.prototype,
      "transferControlToOffscreen",
      {
        configurable: true,
        value: () => ({ width: 0, height: 0 }),
      },
    );
    vi.stubGlobal("Worker", MockWorker);
    let category = { id: 1, name: "object", supercategory: "phytolith", color: "#4094dc", active: true };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      let body: unknown = DATA_TREE;
      if (url === "/api/classes" && (!init?.method || init.method === "GET")) {
        body = { schema_version: 1, next_category_id: 2, categories: [category] };
      } else if (url === "/api/statistics") {
        body = {
          totalAnnotations: 3,
          classes: [{ ...category, annotationCount: 3 }],
          previews: Array.from({ length: 10 }, (_, index) => ({
            imageId: "train-" + index,
            fileName: "train/train-" + index + ".png",
            annotationCount: 3,
            categoryIds: [1],
          })),
        };
      } else if (/\/api\/classes\/1$/.test(url) && init?.method === "PATCH") {
        category = { ...category, ...JSON.parse(String(init.body)) };
        body = category;
      } else if (/\/api\/classes\/1$/.test(url) && init?.method === "DELETE") {
        category = { ...category, active: false };
        body = category;
      } else if (/\/api\/images\/[^/]+\/annotations$/.test(url) && (!init?.method || init.method === "GET")) {
        body = { imageId: "draft", layers: [], latestMaskLayerId: null, preventOverlap: false };
      } else if (/\/api\/images\/[^/]+\/annotations$/.test(url) && init?.method === "PUT") {
        body = { savedLayers: 1, emptyLayers: 0 };
      }
      return { ok: true, status: 200, json: async () => body };
    }));
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: FrameRequestCallback) => {
        queueMicrotask(() => callback(performance.now()));
        return 1;
      },
    );

    await import("./main");
  });

  it("navigates folders and images, then handles point interactions", async () => {
    const worker = MockWorker.instance;
    expect(worker.messages[0]).toMatchObject({ type: "initialize" });

    await vi.waitFor(() => {
      expect(
        document.querySelector("#folder-tree-value")?.textContent,
      ).toBe("Data / train");
    });

    worker.emit({ type: "model-ready" });
    await Promise.resolve();
    await Promise.resolve();
    expect(worker.messages.at(-1)).toMatchObject({
      type: "load-image",
      imageRevision: 1,
      imageId: "train-a",
      url: "/data/train/train-a.png",
    });
    const setupTab = document.querySelector<HTMLButtonElement>("#setup-tab")!;
    const maskingTab = document.querySelector<HTMLButtonElement>("#masking-tab")!;
    const statisticsTab = document.querySelector<HTMLButtonElement>("#statistics-tab")!;
    const statisticsPanel = document.querySelector<HTMLDivElement>("#statistics-panel")!;
    const setupPanel = document.querySelector<HTMLDivElement>("#setup-panel")!;
    const maskingPanel = document.querySelector<HTMLDivElement>("#masking-panel")!;
    expect(maskingTab.getAttribute("aria-selected")).toBe("true");
    expect(maskingPanel.hidden).toBe(false);
    expect(setupPanel.hidden).toBe(true);
    statisticsTab.click();
    await vi.waitFor(() => {
      expect(statisticsPanel.hidden).toBe(false);
      expect(document.querySelector("#statistics-total")?.textContent).toBe("3 saved annotations");
      expect(document.querySelector(".statistics-row strong")?.textContent).toBe("object");
      expect(document.querySelectorAll(".statistics-preview-card")).toHaveLength(8);
    });
    setupTab.click();
    expect(setupPanel.hidden).toBe(false);
    expect(maskingPanel.hidden).toBe(true);

    document.querySelector<HTMLButtonElement>("#next-folder-button")!.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(
      document.querySelector("#folder-tree-value")?.textContent,
    ).toBe("Data / val");
    expect(
      document.querySelector<HTMLSelectElement>("#image-select")?.value,
    ).toBe("val/val-a.png");
    expect(worker.messages.at(-1)).toMatchObject({
      type: "load-image",
      url: "/data/val/val-a.png",
    });
    expect(setupPanel.hidden).toBe(false);
    maskingTab.click();
    expect(maskingPanel.hidden).toBe(false);

    setupTab.click();
    document.querySelector<HTMLButtonElement>("#folder-tree-trigger")!.click();
    const treeMenu =
      document.querySelector<HTMLDivElement>("#folder-tree-menu")!;
    expect(treeMenu.hidden).toBe(false);
    expect(
      treeMenu.querySelector('[data-folder-path="train/nested"]'),
    ).not.toBeNull();
    treeMenu
      .querySelector<HTMLButtonElement>('[data-folder-path="train"]')!
      .click();
    await Promise.resolve();
    await Promise.resolve();
    expect(treeMenu.hidden).toBe(true);
    maskingTab.click();

    const imageSelect =
      document.querySelector<HTMLSelectElement>("#image-select")!;
    imageSelect.value = "train/train-b.png";
    imageSelect.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
    await Promise.resolve();
    expect(
      document.querySelector<HTMLButtonElement>("#next-image-button")?.disabled,
    ).toBe(true);

    document.querySelector<HTMLButtonElement>("#previous-image-button")!.click();
    await Promise.resolve();
    await Promise.resolve();
    expect(imageSelect.value).toBe("train/train-a.png");

    const latestLoad = [...worker.messages]
      .reverse()
      .find(
        (
          message,
        ): message is Extract<
          MainToWorkerMessage,
          { type: "load-image" }
        > => message.type === "load-image",
      )!;
    worker.emit({
      type: "image-ready",
      imageRevision: latestLoad.imageRevision,
      width: 982,
      height: 996,
      encodeMs: 1500,
      cacheHit: true,
    });

    const stage = document.querySelector<HTMLDivElement>("#image-stage")!;
    expect(stage.style.aspectRatio).toBe("982 / 996");
    vi.spyOn(stage, "getBoundingClientRect").mockReturnValue({
      x: 10,
      y: 20,
      left: 10,
      top: 20,
      right: 210,
      bottom: 120,
      width: 200,
      height: 100,
      toJSON: () => ({}),
    });

    stage.dispatchEvent(
      new MouseEvent("pointerenter", {
        bubbles: true,
        clientX: 110,
        clientY: 70,
      }),
    );
    await Promise.resolve();

    expect(worker.messages.at(-1)).toMatchObject({
      type: "decode",
      points: [{ x: 0.5, y: 0.5, label: 1 }],
    });

    document.querySelector<HTMLButtonElement>("#negative-tool")!.click();
    expect(worker.messages.at(-1)).toMatchObject({
      type: "decode",
      points: [{ x: 0.5, y: 0.5, label: 0 }],
    });

    stage.dispatchEvent(
      new MouseEvent("click", {
        bubbles: true,
        clientX: 110,
        clientY: 70,
      }),
    );
    expect(document.querySelector("#point-count")?.textContent).toBe(
      "1 pinned",
    );
    expect(worker.messages.at(-1)).toMatchObject({
      type: "decode",
      points: [{ x: 0.5, y: 0.5, label: 0 }],
    });
    const saveButton = document.querySelector<HTMLButtonElement>("#save-annotations")!;
    const autosaveButton = document.querySelector<HTMLButtonElement>("#autosave-toggle")!;
    expect(saveButton.textContent).toBe("Save");
    expect(autosaveButton.getAttribute("aria-pressed")).toBe("true");
    autosaveButton.click();
    expect(autosaveButton.getAttribute("aria-pressed")).toBe("false");
    saveButton.click();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(vi.mocked(fetch).mock.calls.some(([url, init]) =>
      String(url).includes("/annotations") && init?.method === "PUT"
    )).toBe(true);
    expect(saveButton.textContent).toBe("Saved");

    document.querySelector<HTMLButtonElement>("#undo-button")!.click();
    expect(document.querySelector("#point-count")?.textContent).toBe(
      "0 pinned",
    );
    expect(worker.messages.at(-1)).toMatchObject({ type: "clear" });

    stage.dispatchEvent(
      new MouseEvent("pointermove", {
        bubbles: true,
        clientX: 60,
        clientY: 45,
      }),
    );
    await Promise.resolve();
    stage.dispatchEvent(
      new MouseEvent("click", {
        bubbles: true,
        clientX: 60,
        clientY: 45,
      }),
    );
    document.querySelector<HTMLButtonElement>("#clear-button")!.click();

    expect(document.querySelector("#point-count")?.textContent).toBe(
      "0 pinned",
    );
    expect(worker.messages.at(-1)).toMatchObject({ type: "clear" });

    worker.emit({
      type: "edit-state",
      imageRevision: latestLoad.imageRevision,
      layerId: latestLoad.activeLayerId,
      editRevision: 0,
      hasMask: true,
      hasEdits: false,
      canUndo: false,
      inverted: false,
    });
    const markerTool = document.querySelector<HTMLButtonElement>("#marker-tool")!;
    const eraserTool = document.querySelector<HTMLButtonElement>("#eraser-tool")!;
    const markerSize = document.querySelector<HTMLInputElement>("#marker-size")!;
    const eraserSize = document.querySelector<HTMLInputElement>("#eraser-size")!;
    expect(markerTool.disabled).toBe(false);

    markerSize.value = "40";
    markerSize.dispatchEvent(new Event("input", { bubbles: true }));
    markerTool.click();
    expect(markerTool.getAttribute("aria-pressed")).toBe("true");

    const messagesBeforeHover = worker.messages.length;
    stage.dispatchEvent(pointerEvent("pointerenter", 9, 110, 70));
    stage.dispatchEvent(pointerEvent("pointermove", 9, 120, 70));
    expect(worker.messages).toHaveLength(messagesBeforeHover);
    expect(
      document.querySelector<HTMLDivElement>("#brush-cursor")!.style.width,
    ).toBe(`${40 * (200 / 982)}px`);

    const setPointerCapture = vi.fn();
    const releasePointerCapture = vi.fn();
    Object.defineProperty(stage, "setPointerCapture", { value: setPointerCapture });
    Object.defineProperty(stage, "releasePointerCapture", { value: releasePointerCapture });
    stage.dispatchEvent(pointerEvent("pointerdown", 9, 110, 70));
    expect(setPointerCapture).toHaveBeenCalledWith(9);
    expect(worker.messages.at(-1)).toMatchObject({
      type: "brush",
      phase: "begin",
      operation: "add",
      radius: 20,
      points: [{ x: 0.5, y: 0.5 }],
    });
    stage.dispatchEvent(pointerEvent("pointermove", 9, 160, 70));
    expect(worker.messages.at(-1)).toMatchObject({
      type: "brush",
      phase: "continue",
    });
    stage.dispatchEvent(pointerEvent("pointerup", 9, 160, 70));
    expect(worker.messages.at(-1)).toMatchObject({
      type: "brush",
      phase: "end",
    });
    expect(releasePointerCapture).toHaveBeenCalledWith(9);

    worker.emit({
      type: "edit-state",
      imageRevision: latestLoad.imageRevision,
      layerId: latestLoad.activeLayerId,
      editRevision: 1,
      hasMask: true,
      hasEdits: true,
      canUndo: true,
      inverted: false,
    });
    document.querySelector<HTMLButtonElement>("#invert-mask")!.click();
    expect(worker.messages.at(-1)).toMatchObject({ type: "invert-mask" });
    worker.emit({
      type: "edit-state",
      imageRevision: latestLoad.imageRevision,
      layerId: latestLoad.activeLayerId,
      editRevision: 2,
      hasMask: true,
      hasEdits: true,
      canUndo: true,
      inverted: true,
    });
    expect(
      document.querySelector<HTMLButtonElement>("#invert-mask")!
        .getAttribute("aria-pressed"),
    ).toBe("true");
    document.querySelector<HTMLButtonElement>("#undo-edit")!.click();
    expect(worker.messages.at(-1)).toMatchObject({ type: "undo-edit" });
    document.querySelector<HTMLButtonElement>("#reset-edits")!.click();
    expect(worker.messages.at(-1)).toMatchObject({ type: "reset-edits" });

    eraserSize.value = "60";
    eraserSize.dispatchEvent(new Event("input", { bubbles: true }));
    eraserTool.click();
    stage.dispatchEvent(pointerEvent("pointerdown", 10, 110, 70));
    expect(worker.messages.at(-1)).toMatchObject({
      type: "brush",
      phase: "begin",
      operation: "erase",
      radius: 30,
    });
    expect(markerSize.value).toBe("40");
    stage.dispatchEvent(pointerEvent("pointerup", 10, 110, 70));

    document.querySelector<HTMLButtonElement>("#add-mask")!.click();
    const created = [...worker.messages].reverse().find(
      (message) => message.type === "create-layer",
    );
    expect(created).toMatchObject({ type: "create-layer", layer: { visible: true } });
    if (!created || created.type !== "create-layer") throw new Error("Layer was not created.");
    expect(document.querySelectorAll(".mask-layer-row")).toHaveLength(2);
    expect(document.querySelector("#mask-count")?.textContent).toBe("2 masks");

    const eye = document.querySelector<HTMLButtonElement>(
      `[data-layer-visible="${created.layer.id}"]`,
    )!;
    eye.click();
    expect(worker.messages.at(-1)).toMatchObject({
      type: "update-layer",
      layerId: created.layer.id,
      visible: false,
    });
    expect(markerTool.disabled).toBe(true);
    const messagesWhileHidden = worker.messages.length;
    stage.dispatchEvent(pointerEvent("pointerenter", 11, 110, 70));
    stage.dispatchEvent(pointerEvent("pointermove", 11, 120, 70));
    stage.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: 120, clientY: 70 }));
    await Promise.resolve();
    expect(worker.messages).toHaveLength(messagesWhileHidden);

    eye.click();
    const name = document.querySelector<HTMLInputElement>("#mask-layer-name")!;
    name.value = "Class A";
    name.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
    await Promise.resolve();
    expect(
      document.querySelector(".mask-layer-row.active .layer-select span:last-child")?.textContent,
    ).toBe("Class A · 2");
    const color = document.querySelector<HTMLInputElement>("#mask-layer-color")!;
    color.value = "#112233";
    color.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
    await Promise.resolve();
    expect(worker.messages.at(-1)).toMatchObject({
      type: "update-layer",
      layerId: created.layer.id,
      color: "#112233",
    });
    document.querySelector<HTMLButtonElement>("#delete-mask")!.click();
    expect(worker.messages.some((message) => message.type === "delete-layer" && message.layerId === created.layer.id)).toBe(true);
    expect(document.querySelectorAll(".mask-layer-row")).toHaveLength(1);
    expect(document.querySelector<HTMLButtonElement>("#delete-mask")!.disabled).toBe(true);

    const preventOverlap = document.querySelector<HTMLInputElement>("#prevent-mask-overlap")!;
    preventOverlap.checked = true;
    preventOverlap.dispatchEvent(new Event("change", { bubbles: true }));
    expect(worker.messages.at(-1)).toMatchObject({
      type: "set-overlap-prevention",
      enabled: true,
    });
  });
});

function pointerEvent(
  type: string,
  pointerId: number,
  clientX: number,
  clientY: number,
): MouseEvent {
  const event = new MouseEvent(type, {
    bubbles: true,
    button: 0,
    clientX,
    clientY,
  });
  Object.defineProperty(event, "pointerId", { value: pointerId });
  Object.defineProperty(event, "getCoalescedEvents", {
    value: () => [event],
  });
  return event;
}
