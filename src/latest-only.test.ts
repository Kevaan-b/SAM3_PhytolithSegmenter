import { describe, expect, it } from "vitest";
import { LatestOnlyRunner } from "./latest-only";

describe("LatestOnlyRunner", () => {
  it("runs one task at a time and keeps only the newest pending value", async () => {
    const started: number[] = [];
    const finished: number[] = [];
    const releases: Array<() => void> = [];

    const runner = new LatestOnlyRunner<number>(async (value) => {
      started.push(value);
      await new Promise<void>((resolve) => releases.push(resolve));
      finished.push(value);
    });

    runner.push(1);
    runner.push(2);
    runner.push(3);
    await Promise.resolve();

    expect(started).toEqual([1]);
    releases.shift()?.();
    await Promise.resolve();
    await Promise.resolve();

    expect(started).toEqual([1, 3]);
    releases.shift()?.();
    await runner.whenIdle();

    expect(finished).toEqual([1, 3]);
    expect(runner.isRunning()).toBe(false);
  });

  it("can discard pending work without interrupting the active task", async () => {
    const seen: string[] = [];
    let release = () => {};
    const runner = new LatestOnlyRunner<string>(async (value) => {
      seen.push(value);
      await new Promise<void>((resolve) => {
        release = resolve;
      });
    });

    runner.push("active");
    runner.push("stale");
    runner.clearPending();
    release();
    await runner.whenIdle();

    expect(seen).toEqual(["active"]);
  });
});
