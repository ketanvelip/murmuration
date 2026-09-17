import flockWgsl from "../shaders/flock.wgsl?raw";
import { SpatialSort } from "./spatial";
import type { PassTimer } from "../gpu/timing";

export interface FlockConfig {
  agents: number;
  gridX: number;
  gridY: number;
  cellSize: number;
}

export interface Dynamics {
  dt: number;
  roost: [number, number];
  predator: [number, number];
  predatorWeight: number;
  predatorRadius: number;
  maxSpeed: number;
  minSpeed: number;
}

const TUNING = {
  sepW: 1.75,
  aliW: 1.15,
  cohW: 0.72,
  roostW: 0.2,
  percept: 26,
  sepDist: 13,
  maxForce: 1250,
  edge: 54,
};

/**
 * The flock: positions, velocities, spatial index, and the steering pass.
 *
 * Positions and velocities are double-buffered because neighbours read each
 * other within a single dispatch - writing in place would have birds steering
 * off half-updated data.
 */
export class Flock {
  readonly cfg: FlockConfig;
  readonly spatial: SpatialSort;
  readonly posPair: [GPUBuffer, GPUBuffer];
  readonly velPair: [GPUBuffer, GPUBuffer];
  readonly alive: GPUBuffer;
  readonly world: [number, number];

  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroups: GPUBindGroup[];
  private parity = 0;

  constructor(device: GPUDevice, cfg: FlockConfig) {
    this.device = device;
    this.cfg = cfg;
    this.world = [cfg.gridX * cfg.cellSize, cfg.gridY * cfg.cellSize];

    const S = GPUBufferUsage.STORAGE;
    const DST = GPUBufferUsage.COPY_DST;
    const SRC = GPUBufferUsage.COPY_SRC;

    const mk = (label: string, bytes: number) =>
      device.createBuffer({ size: bytes, usage: S | DST | SRC, label });

    this.posPair = [mk("posA", cfg.agents * 8), mk("posB", cfg.agents * 8)];
    this.velPair = [mk("velA", cfg.agents * 8), mk("velB", cfg.agents * 8)];
    this.alive = mk("alive", cfg.agents * 4);

    this.spatial = new SpatialSort(
      device,
      { agents: cfg.agents, gridX: cfg.gridX, gridY: cfg.gridY, cellSize: cfg.cellSize },
      this.posPair,
    );

    this.params = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | DST,
      label: "params:flock",
    });

    this.pipeline = device.createComputePipeline({
      label: "flock",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: flockWgsl, label: "flock" }),
        entryPoint: "main",
      },
    });

    const bg = (inIdx: number) =>
      device.createBindGroup({
        label: `bg:flock${inIdx}`,
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: this.posPair[inIdx] as GPUBuffer } },
          { binding: 2, resource: { buffer: this.velPair[inIdx] as GPUBuffer } },
          { binding: 3, resource: { buffer: this.posPair[1 - inIdx] as GPUBuffer } },
          { binding: 4, resource: { buffer: this.velPair[1 - inIdx] as GPUBuffer } },
          { binding: 5, resource: { buffer: this.spatial.offsets } },
          { binding: 6, resource: { buffer: this.spatial.sortedIdx } },
        ],
      });

    this.bindGroups = [bg(0), bg(1)];
    this.seed();
  }

  /** Scatter the flock into loose sub-flocks, all already in motion. */
  seed(): void {
    const { agents } = this.cfg;
    const [wx, wy] = this.world;
    const pos = new Float32Array(agents * 2);
    const vel = new Float32Array(agents * 2);
    const groups = 5;

    for (let i = 0; i < agents; i++) {
      const g = i % groups;
      const gx = wx * (0.22 + 0.14 * g);
      const gy = wy * (0.35 + 0.12 * (g % 3));
      const a = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * Math.min(wx, wy) * 0.17;
      pos[i * 2] = Math.min(wx - 1, Math.max(1, gx + Math.cos(a) * r));
      pos[i * 2 + 1] = Math.min(wy - 1, Math.max(1, gy + Math.sin(a) * r));
      const va = a + 1.4;
      vel[i * 2] = Math.cos(va) * 130;
      vel[i * 2 + 1] = Math.sin(va) * 130;
    }

    const q = this.device.queue;
    for (const b of this.posPair) q.writeBuffer(b, 0, pos.buffer as ArrayBuffer, 0, pos.byteLength);
    for (const b of this.velPair) q.writeBuffer(b, 0, vel.buffer as ArrayBuffer, 0, vel.byteLength);
    const ones = new Uint32Array(agents).fill(1);
    q.writeBuffer(this.alive, 0, ones.buffer as ArrayBuffer, 0, ones.byteLength);
    this.parity = 0;
  }

  setDynamics(d: Dynamics): void {
    const buf = new ArrayBuffer(96);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    const { cfg } = this;

    u[0] = cfg.agents;
    u[1] = cfg.gridX;
    u[2] = cfg.gridY;
    u[3] = cfg.gridX * cfg.gridY;

    f[4] = cfg.cellSize;
    f[5] = this.world[0];
    f[6] = this.world[1];
    f[7] = d.dt;

    f[8] = d.roost[0];
    f[9] = d.roost[1];
    f[10] = d.maxSpeed;
    f[11] = d.minSpeed;

    f[12] = TUNING.sepW;
    f[13] = TUNING.aliW;
    f[14] = TUNING.cohW;
    f[15] = TUNING.roostW;

    f[16] = TUNING.percept * TUNING.percept;
    f[17] = TUNING.sepDist * TUNING.sepDist;
    f[18] = TUNING.maxForce;
    f[19] = TUNING.edge;

    f[20] = d.predator[0];
    f[21] = d.predator[1];
    f[22] = d.predatorWeight;
    f[23] = d.predatorRadius * d.predatorRadius;

    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Rebin, then steer. Flips parity, so `currentPos` is the new state. */
  record(encoder: GPUCommandEncoder, timer?: PassTimer): void {
    this.spatial.record(encoder, timer, this.parity);

    const writes = timer?.slot("flock");
    const pass = encoder.beginComputePass(
      writes ? { label: "flock", timestampWrites: writes } : { label: "flock" },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[this.parity] as GPUBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.cfg.agents / 256));
    pass.end();

    this.parity = 1 - this.parity;
  }

  /** Index of the buffer holding the current state. */
  get current(): number {
    return this.parity;
  }
}
