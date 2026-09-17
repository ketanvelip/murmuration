/**
 * Non-blocking GPU→CPU readback over a ring of staging buffers.
 *
 * A buffer cannot be mapped while the GPU is using it, so a single staging
 * buffer would force a stall. Instead we keep a small pool: each frame takes a
 * free buffer, copies into it, and kicks off mapAsync. Whichever resolves most
 * recently becomes the value the CPU reads.
 *
 * If every buffer is in flight the frame simply skips its copy rather than
 * blocking - a dropped sample is always better than a pipeline stall.
 */
export class Readback {
  private readonly size: number;
  private readonly free: GPUBuffer[] = [];
  private readonly pending: GPUBuffer[] = [];
  private readonly all: GPUBuffer[] = [];

  private latestBytes: ArrayBuffer | null = null;
  private readonly recordedAt = new Map<GPUBuffer, number>();

  /** Frames where the ring was exhausted and no copy was recorded. */
  skipped = 0;
  /** Samples successfully mapped back. */
  received = 0;
  /**
   * Wall-clock milliseconds between recording a copy and the CPU being able to
   * read it. This is the pace-independent number: latency counted in frames
   * depends entirely on how fast the loop happens to be spinning.
   */
  readonly latenciesMs: number[] = [];

  constructor(device: GPUDevice, size: number, depth = 3) {
    this.size = size;
    for (let i = 0; i < depth; i++) {
      const b = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: `readback[${i}]`,
      });
      this.free.push(b);
      this.all.push(b);
    }
  }

  /** Record a copy into a free staging buffer. No-op when all are in flight. */
  record(encoder: GPUCommandEncoder, src: GPUBuffer): void {
    const buf = this.free.pop();
    if (!buf) {
      this.skipped++;
      return;
    }
    encoder.copyBufferToBuffer(src, 0, buf, 0, this.size);
    this.recordedAt.set(buf, performance.now());
    this.pending.push(buf);
  }

  /** Call once after submitting the encoder that `record` was called on. */
  kick(): void {
    while (this.pending.length > 0) {
      const buf = this.pending.pop();
      if (!buf) break;
      buf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const at = this.recordedAt.get(buf);
          if (at !== undefined) this.latenciesMs.push(performance.now() - at);
          this.latestBytes = buf.getMappedRange().slice(0);
          buf.unmap();
          this.received++;
          this.free.push(buf);
        })
        .catch(() => {
          // Device lost or buffer destroyed mid-flight; drop the sample.
          this.free.push(buf);
        });
    }
  }

  /** Most recent sample the GPU has handed back, or null if none yet. */
  latest(): Uint32Array | null {
    return this.latestBytes ? new Uint32Array(this.latestBytes) : null;
  }

  destroy(): void {
    for (const b of this.all) b.destroy();
  }
}
