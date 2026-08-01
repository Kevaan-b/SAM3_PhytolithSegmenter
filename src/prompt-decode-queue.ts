export class PromptDecodeQueue<T extends { preview: boolean }> {
  private committed: T | undefined;
  private preview: T | undefined;

  push(value: T): void {
    if (value.preview) {
      this.preview = value;
    } else {
      this.committed = value;
      this.preview = undefined;
    }
  }

  take(): T | undefined {
    if (this.committed) {
      const value = this.committed;
      this.committed = undefined;
      return value;
    }
    const value = this.preview;
    this.preview = undefined;
    return value;
  }

  clear(): void {
    this.committed = undefined;
    this.preview = undefined;
  }

  clearPreview(matches: (value: T) => boolean = () => true): void {
    if (this.preview && matches(this.preview)) this.preview = undefined;
  }

  clearMatching(matches: (value: T) => boolean): void {
    if (this.committed && matches(this.committed)) this.committed = undefined;
    if (this.preview && matches(this.preview)) this.preview = undefined;
  }

  hasPending(): boolean {
    return this.committed !== undefined || this.preview !== undefined;
  }
}
