import flockWgsl from "../shaders/flock.wgsl?raw";
import reorderWgsl from "../shaders/reorder.wgsl?raw";
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
  blast: [number, number];
  blastWeight: number;
  blastRadius: number;
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
 * The flock: agent state, spatial index, reorder, and the steering pass.
 *
 * Two sets of buffers, but not a conventional double buffer - they have fixed
 * roles:
 *
 *   pos/vel/alive              canonical state, in whatever order it ended up
 *   posSorted/velSorted/...    the same agents gathered into cell order
 *
 * Each frame bins `pos`, gathers into the sorted set, steers reading the sorted
 * set, and writes results back into the canonical set. Because the steering
 * pass reads cell-ordered data, a bird's neighbours are contiguous in memory
 * and offsets[] indexes straight into the arrays.
 *
 * Agents are effectively renumbered every frame. Nothing outside holds a bird's
 * identity across frames, so that costs nothing - but it does mean this cannot
 * carry per-bird state that the CPU tracks by index.
 */
export class Flock {
  readonly cfg: FlockConfig;
  readonly spatial: SpatialSort;
  readonly world: [number, number];

  readonly pos: GPUBuffer;
  readonly vel: GPUBuffer;
  readonly alive: GPUBuffer;

  private readonly posSorted: GPUBuffer;
  private readonly velSorted: GPUBuffer;
  private readonly aliveSorted: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly reorderParams: GPUBuffer;
  private readonly flockPipe: GPUComputePipeline;
  private readonly reorderPipe: GPUComputePipeline;
  private readonly bgFlock: GPUBindGroup;
  private readonly bgReorder: GPUBindGroup;
  private frame = 0;

  constructor(device: GPUDevice, cfg: FlockConfig) {
    this.device = device;
    this.cfg = cfg;
    this.world = [cfg.gridX * cfg.cellSize, cfg.gridY * cfg.cellSize];

    const S = GPUBufferUsage.STORAGE;
    const DST = GPUBufferUsage.COPY_DST;
    const SRC = GPUBufferUsage.COPY_SRC;
    const mk = (label: string, bytes: number) =>
      device.createBuffer({ size: bytes, usage: S | DST | SRC, label });

    this.pos = mk("pos", cfg.agents * 8);
    this.vel = mk("vel", cfg.agents * 8);
    this.alive = mk("alive", cfg.agents * 4);
    this.posSorted = mk("posSorted", cfg.agents * 8);
    this.velSorted = mk("velSorted", cfg.agents * 8);
    this.aliveSorted = mk("aliveSorted", cfg.agents * 4);

    this.spatial = new SpatialSort(
      device,
      { agents: cfg.agents, gridX: cfg.gridX, gridY: cfg.gridY, cellSize: cfg.cellSize },
      this.pos,
    );

    this.params = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | DST,
      label: "params:flock",
    });
    this.reorderParams = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | DST,
      label: "params:reorder",
    });
    device.queue.writeBuffer(this.reorderParams, 0, new Uint32Array([cfg.agents, 0, 0, 0]));

    const pipe = (code: string, label: string) =>
      device.createComputePipeline({
        label,
        layout: "auto",
        compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" },
      });

    this.reorderPipe = pipe(reorderWgsl, "reorder");
    this.flockPipe = pipe(flockWgsl, "flock");

    const bind = (p: GPUComputePipeline, res: GPUBuffer[], label: string) =>
      device.createBindGroup({
        label,
        layout: p.getBindGroupLayout(0),
        entries: res.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

    this.bgReorder = bind(
      this.reorderPipe,
      [
        this.reorderParams,
        this.spatial.sortedIdx,
        this.pos,
        this.vel,
        this.alive,
        this.posSorted,
        this.velSorted,
        this.aliveSorted,
      ],
      "bg:reorder",
    );

    this.bgFlock = bind(
      this.flockPipe,
      [
        this.params,
        this.posSorted,
        this.velSorted,
        this.pos,
        this.vel,
        this.spatial.offsets,
        this.aliveSorted,
        this.alive,
      ],
      "bg:flock",
    );

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
    q.writeBuffer(this.pos, 0, pos.buffer as ArrayBuffer, 0, pos.byteLength);
    q.writeBuffer(this.vel, 0, vel.buffer as ArrayBuffer, 0, vel.byteLength);
    const ones = new Uint32Array(agents).fill(1);
    q.writeBuffer(this.alive, 0, ones.buffer as ArrayBuffer, 0, ones.byteLength);
  }

  setDynamics(d: Dynamics): void {
    const buf = new ArrayBuffer(128);
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

    u[24] = this.frame >>> 0;

    f[28] = d.blast[0];
    f[29] = d.blast[1];
    f[30] = d.blastWeight;
    f[31] = d.blastRadius * d.blastRadius;

    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /**
   * Build the spatial index over the current positions.
   *
   * Split from the steering pass so gameplay collision can run between the
   * two: it needs an index that still describes where the birds are, and
   * `recordSteer` is what moves them.
   */
  recordIndex(encoder: GPUCommandEncoder, timer?: PassTimer): void {
    this.spatial.record(encoder, timer);
  }

  /** Gather into cell order, then steer. */
  recordSteer(encoder: GPUCommandEncoder, timer?: PassTimer): void {
    const groups = Math.ceil(this.cfg.agents / 256);
    const run = (
      pipeline: GPUComputePipeline,
      group: GPUBindGroup,
      label: string,
    ) => {
      const writes = timer?.slot(label);
      const pass = encoder.beginComputePass(writes ? { label, timestampWrites: writes } : { label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
      pass.end();
    };

    run(this.reorderPipe, this.bgReorder, "reorder");
    run(this.flockPipe, this.bgFlock, "flock");
    this.frame++;
  }

  /** Index and steer in one go, for callers with no collision to interleave. */
  record(encoder: GPUCommandEncoder, timer?: PassTimer): void {
    this.recordIndex(encoder, timer);
    this.recordSteer(encoder, timer);
  }
}
