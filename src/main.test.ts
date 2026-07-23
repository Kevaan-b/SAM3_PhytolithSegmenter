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
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => DATA_TREE,
      })),
    );
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
    expect(worker.messages.at(-1)).toMatchObject({
      type: "load-image",
      imageRevision: 1,
      imageId: "train-a",
      url: "/data/train/train-a.png",
    });

    document.querySelector<HTMLButtonElement>("#next-button")!.click();
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
    expect(treeMenu.hidden).toBe(true);

    const imageSelect =
      document.querySelector<HTMLSelectElement>("#image-select")!;
    imageSelect.value = "train/train-b.png";
    imageSelect.dispatchEvent(new Event("change", { bubbles: true }));
    expect(
      document.querySelector<HTMLButtonElement>("#navigate-folders")
        ?.getAttribute("aria-pressed"),
    ).toBe("true");

    document.querySelector<HTMLButtonElement>("#navigate-images")!.click();
    expect(
      document.querySelector<HTMLButtonElement>("#navigate-images")
        ?.getAttribute("aria-pressed"),
    ).toBe("true");
    expect(
      document.querySelector<HTMLButtonElement>("#next-button")?.disabled,
    ).toBe(true);

    document.querySelector<HTMLButtonElement>("#previous-button")!.click();
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
  });
});
