import collideWgsl from "../shaders/collide.wgsl?raw";
import type { PassTimer } from "../gpu/timing";
import type { SpatialSort } from "./spatial";

/** Bytes per probe: vec2 pos, f32 radius, f32 kind. */
export const PROBE_STRIDE = 16;
/** Bytes in the status block: kills, playerHits, frame, pad. */
export const STATUS_SIZE = 16;

export const STATUS_KILLS = 0;
export const STATUS_PLAYER_HITS = 1;
export const STATUS_FRAME = 2;

/**
 * Gameplay collision against agents that live in GPU buffers.
 *
 * Probes (pulse rounds, the player hull) are CPU-authoritative and few, so they
 * are uploaded each frame. Everything they hit is resolved on the GPU; only the
 * 16-byte status block travels back.
 */
export class Collider {
  readonly maxProbes: number;
  readonly probes: GPUBuffer;
  readonly alive: GPUBuffer;
  readonly status: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly agents: number;

  constructor(device: GPUDevice, spatial: SpatialSort, maxProbes = 128) {
    this.device = device;
    this.maxProbes = maxProbes;
    this.agents = spatial.cfg.agents;

    const S = GPUBufferUsage.STORAGE;
    const SRC = GPUBufferUsage.COPY_SRC;
    const DST = GPUBufferUsage.COPY_DST;

    this.probes = device.createBuffer({
      size: maxProbes * PROBE_STRIDE,
      usage: S | DST,
      label: "probes",
    });
    this.alive = device.createBuffer({
      size: this.agents * 4,
      usage: S | SRC | DST,
      label: "alive",
    });
    this.status = device.createBuffer({
      size: STATUS_SIZE,
      usage: S | SRC | DST,
      label: "status",
    });
    this.params = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | DST,
      label: "params:collide",
    });

    const cfg = spatial.cfg;
    const p = new ArrayBuffer(32);
    new Uint32Array(p, 0, 4).set([cfg.gridX, cfg.gridY, spatial.cellCount, 0]);
    new Float32Array(p, 16, 3).set([
      cfg.cellSize,
      cfg.gridX * cfg.cellSize,
      cfg.gridY * cfg.cellSize,
    ]);
    device.queue.writeBuffer(this.params, 0, p);

    this.pipeline = device.createComputePipeline({
      label: "collide",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: collideWgsl, label: "collide" }),
        entryPoint: "main",
      },
    });

    this.bindGroup = device.createBindGroup({
      label: "bg:collide",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.probes } },
        { binding: 2, resource: { buffer: spatial.pos } },
        { binding: 3, resource: { buffer: spatial.offsets } },
        { binding: 4, resource: { buffer: spatial.sortedIdx } },
        { binding: 5, resource: { buffer: this.alive } },
        { binding: 6, resource: { buffer: this.status } },
      ],
    });

    this.resetAlive();
  }

  /** Bring every agent back to life. */
  resetAlive(): void {
    const ones = new Uint32Array(this.agents).fill(1);
    this.device.queue.writeBuffer(this.alive, 0, ones.buffer as ArrayBuffer, 0, ones.byteLength);
  }

  /** Upload probes and tell the shader how many are live this frame. */
  writeProbes(data: Float32Array, count: number): void {
    this.device.queue.writeBuffer(
      this.probes,
      0,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      Math.min(count, this.maxProbes) * PROBE_STRIDE,
    );
    this.device.queue.writeBuffer(this.params, 12, new Uint32Array([Math.min(count, this.maxProbes)]));
  }

  /**
   * Zero the counters and stamp the frame number. The stamp is what makes
   * readback latency measurable: whatever the CPU eventually reads back says
   * which frame produced it.
   */
  resetStatus(frame: number): void {
    this.device.queue.writeBuffer(this.status, 0, new Uint32Array([0, 0, frame >>> 0, 0]));
  }

  record(encoder: GPUCommandEncoder, probeCount: number, timer?: PassTimer): void {
    const groups = Math.max(1, Math.min(probeCount, this.maxProbes));
    const writes = timer?.slot("collide");
    const pass = encoder.beginComputePass(
      writes ? { label: "collide", timestampWrites: writes } : { label: "collide" },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  destroy(): void {
    this.probes.destroy();
    this.alive.destroy();
    this.status.destroy();
    this.params.destroy();
  }
}
