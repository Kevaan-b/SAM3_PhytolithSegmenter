// @vitest-environment jsdom

import { beforeAll, describe, expect, it, vi } from "vitest";
import type {
  MainToWorkerMessage,
  WorkerToMainMessage,
} from "./protocol";

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
      "requestAnimationFrame",
      (callback: FrameRequestCallback) => {
        queueMicrotask(() => callback(performance.now()));
        return 1;
      },
    );

    await import("./main");
  });

  it("loads an example and handles hover, tools, pinning, undo, and clear", async () => {
    const worker = MockWorker.instance;
    expect(worker.messages[0]).toMatchObject({ type: "initialize" });

    worker.emit({ type: "model-ready" });
    expect(worker.messages.at(-1)).toMatchObject({
      type: "load-image",
      imageRevision: 0,
    });

    worker.emit({
      type: "image-ready",
      imageRevision: 0,
      width: 982,
      height: 996,
      encodeMs: 1500,
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
