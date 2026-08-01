import { describe, expect, it } from "vitest";
import { PromptDecodeQueue } from "./prompt-decode-queue";

interface Job { id: string; preview: boolean }

describe("PromptDecodeQueue", () => {
  it("protects a click commit from newer hover traffic", () => {
    const queue = new PromptDecodeQueue<Job>();
    queue.push({ id: "hover-running", preview: true });
    expect(queue.take()?.id).toBe("hover-running");

    queue.push({ id: "click", preview: false });
    queue.push({ id: "hover-1", preview: true });
    queue.push({ id: "hover-2", preview: true });

    expect(queue.take()?.id).toBe("click");
    expect(queue.take()?.id).toBe("hover-2");
    expect(queue.hasPending()).toBe(false);
  });

  it("keeps only the newest click and drops previews older than that click", () => {
    const queue = new PromptDecodeQueue<Job>();
    queue.push({ id: "hover", preview: true });
    queue.push({ id: "click-1", preview: false });
    queue.push({ id: "click-2", preview: false });

    expect(queue.take()?.id).toBe("click-2");
    expect(queue.take()).toBeUndefined();
  });

  it("can clear one layer's preview without dropping a protected commit", () => {
    const queue = new PromptDecodeQueue<Job>();
    queue.push({ id: "commit-a", preview: false });
    queue.push({ id: "preview-a", preview: true });
    queue.clearPreview((job) => job.id.endsWith("-a"));

    expect(queue.take()?.id).toBe("commit-a");
  });
});
