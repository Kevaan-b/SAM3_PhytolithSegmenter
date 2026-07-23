export class LatestOnlyRunner<T> {
  private pending: T | undefined;
  private running = false;
  private idleResolvers: Array<() => void> = [];

  constructor(private readonly run: (value: T) => Promise<void>) {}

  push(value: T): void {
    this.pending = value;
    void this.drain();
  }

  clearPending(): void {
    this.pending = undefined;
  }

  isRunning(): boolean {
    return this.running;
  }

  async whenIdle(): Promise<void> {
    if (!this.running && this.pending === undefined) return;
    await new Promise<void>((resolve) => this.idleResolvers.push(resolve));
  }

  private async drain(): Promise<void> {
    if (this.running) return;
    this.running = true;

    try {
      while (this.pending !== undefined) {
        const next = this.pending;
        this.pending = undefined;
        await this.run(next);
      }
    } finally {
      this.running = false;
      const resolvers = this.idleResolvers.splice(0);
      resolvers.forEach((resolve) => resolve());
    }
  }
}
