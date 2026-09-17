/**
 * GPU pass timing via timestamp-query.
 *
 * This measures GPU execution time per compute pass, which does not depend on
 * requestAnimationFrame - so it works in a backgrounded tab, where rAF is
 * throttled to near zero. Presented frame rate still needs a foreground window.
 */
export class PassTimer {
  private readonly querySet: GPUQuerySet | null;
  private readonly resolveBuf: GPUBuffer | null;
  private readonly readBuf: GPUBuffer | null;
  private readonly capacity: number;
  private labels: string[] = [];
  private cursor = 0;

  readonly enabled: boolean;

  constructor(device: GPUDevice, enabled: boolean, maxPasses = 16) {
    this.enabled = enabled;
    this.capacity = maxPasses;

    if (!enabled) {
      this.querySet = null;
      this.resolveBuf = null;
      this.readBuf = null;
      return;
    }

    const bytes = maxPasses * 2 * 8; // two u64 timestamps per pass
    this.querySet = device.createQuerySet({ type: "timestamp", count: maxPasses * 2 });
    this.resolveBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /** Start a fresh set of measurements. */
  begin(): void {
    this.labels = [];
    this.cursor = 0;
  }

  /** Timestamp writes to attach to a compute pass descriptor, or undefined. */
  slot(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled || !this.querySet || this.cursor >= this.capacity) return undefined;
    const i = this.cursor++;
    this.labels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: i * 2,
      endOfPassWriteIndex: i * 2 + 1,
    };
  }

  /** Must be called on the same encoder, after all timed passes are recorded. */
  finish(encoder: GPUCommandEncoder): void {
    if (!this.enabled || !this.querySet || !this.resolveBuf || !this.readBuf) return;
    if (this.cursor === 0) return;
    const count = this.cursor * 2;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, count * 8);
  }

  /** Milliseconds per pass, keyed by label. Resolves after the queue drains. */
  async results(): Promise<Record<string, number>> {
    const out: Record<string, number> = {};
    if (!this.enabled || !this.readBuf || this.cursor === 0) return out;

    await this.readBuf.mapAsync(GPUMapMode.READ);
    const raw = new BigUint64Array(this.readBuf.getMappedRange().slice(0));
    this.readBuf.unmap();

    for (let i = 0; i < this.labels.length; i++) {
      const start = raw[i * 2];
      const end = raw[i * 2 + 1];
      const ns = end > start ? Number(end - start) : 0;
      out[this.labels[i]] = ns / 1e6;
    }
    return out;
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuf?.destroy();
    this.readBuf?.destroy();
  }
}
