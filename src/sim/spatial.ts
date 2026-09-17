import histogramWgsl from "../shaders/histogram.wgsl?raw";
import scanWgsl from "../shaders/scan.wgsl?raw";
import addOffsetsWgsl from "../shaders/addOffsets.wgsl?raw";
import scatterWgsl from "../shaders/scatter.wgsl?raw";
import { readBuffer } from "../gpu/device";
import type { PassTimer } from "../gpu/timing";

const WG = 256;
const BLOCK = 1024; // elements scanned per workgroup, must match scan.wgsl

export interface SpatialConfig {
  agents: number;
  gridX: number;
  gridY: number;
  cellSize: number;
}

export interface ValidationReport {
  ok: boolean;
  failures: string[];
  agents: number;
  cells: number;
  occupiedCells: number;
  maxCellOccupancy: number;
  meanOccupancy: number;
}

/**
 * Spatial binning by counting sort, entirely on the GPU.
 *
 *   clearBuffer(counts)        - built in, no dispatch
 *   histogram                  - cell per agent, atomic occupancy counts
 *   scan x2 + addOffsets       - exclusive prefix sum over cells
 *   copyBufferToBuffer         - offsets -> cursor
 *   scatter                    - agent indices into per-cell slots
 *
 * O(n) in the agent count, versus O(n log^2 n) for a bitonic sort over the
 * same keys. Result: sortedIdx holds every agent index grouped by cell, and
 * cell c owns the range [offsets[c], offsets[c+1]).
 */
export class SpatialSort {
  readonly cfg: SpatialConfig;
  readonly cellCount: number;
  readonly numBlocks: number;

  readonly pos: GPUBuffer;
  readonly cellOf: GPUBuffer;
  readonly counts: GPUBuffer;
  readonly offsets: GPUBuffer;
  readonly cursor: GPUBuffer;
  readonly sortedIdx: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly blockSums: GPUBuffer;
  private readonly blockOff: GPUBuffer;
  private readonly blockSums2: GPUBuffer;

  private readonly pHist: GPUBuffer;
  private readonly pScan1: GPUBuffer;
  private readonly pScan2: GPUBuffer;
  private readonly pAdd: GPUBuffer;
  private readonly pScatter: GPUBuffer;

  private readonly histPipe: GPUComputePipeline;
  private readonly scanPipe: GPUComputePipeline;
  private readonly addPipe: GPUComputePipeline;
  private readonly scatterPipe: GPUComputePipeline;

  private readonly bgHist: GPUBindGroup;
  private readonly bgScan1: GPUBindGroup;
  private readonly bgScan2: GPUBindGroup;
  private readonly bgAdd: GPUBindGroup;
  private readonly bgScatter: GPUBindGroup;

  /**
   * @param posBuffer positions to bin. Supplied by the caller when it owns the
   *                  agent state; otherwise the sort allocates its own.
   */
  constructor(device: GPUDevice, cfg: SpatialConfig, posBuffer?: GPUBuffer) {
    this.device = device;
    this.cfg = cfg;
    this.cellCount = cfg.gridX * cfg.gridY;
    this.numBlocks = Math.ceil(this.cellCount / BLOCK);

    if (this.numBlocks > BLOCK) {
      throw new Error(
        `Grid of ${this.cellCount} cells needs ${this.numBlocks} scan blocks, ` +
          `over the ${BLOCK} a two-level scan can cover. Add a third scan level ` +
          `or use a coarser grid.`,
      );
    }

    const S = GPUBufferUsage.STORAGE;
    const SRC = GPUBufferUsage.COPY_SRC;
    const DST = GPUBufferUsage.COPY_DST;
    const U = GPUBufferUsage.UNIFORM;

    const mk = (size: number, usage: number, label: string) =>
      device.createBuffer({ size: Math.max(4, size), usage, label });

    this.pos = posBuffer ?? mk(cfg.agents * 8, S | SRC | DST, "pos");
    this.cellOf = mk(cfg.agents * 4, S | SRC, "cellOf");
    this.counts = mk(this.cellCount * 4, S | SRC | DST, "counts");
    this.offsets = mk((this.cellCount + 1) * 4, S | SRC | DST, "offsets");
    this.cursor = mk(this.cellCount * 4, S | SRC | DST, "cursor");
    this.sortedIdx = mk(cfg.agents * 4, S | SRC, "sortedIdx");
    this.blockSums = mk(this.numBlocks * 4, S | SRC | DST, "blockSums");
    this.blockOff = mk(this.numBlocks * 4, S | SRC | DST, "blockOff");
    this.blockSums2 = mk(4, S | SRC | DST, "blockSums2");

    this.pHist = mk(32, U | DST, "params:histogram");
    this.pScan1 = mk(16, U | DST, "params:scan1");
    this.pScan2 = mk(16, U | DST, "params:scan2");
    this.pAdd = mk(16, U | DST, "params:addOffsets");
    this.pScatter = mk(16, U | DST, "params:scatter");

    const q = device.queue;
    const hist = new ArrayBuffer(32);
    new Uint32Array(hist, 0, 4).set([cfg.agents, this.cellCount, cfg.gridX, cfg.gridY]);
    new Float32Array(hist, 16, 3).set([
      cfg.cellSize,
      cfg.gridX * cfg.cellSize,
      cfg.gridY * cfg.cellSize,
    ]);
    q.writeBuffer(this.pHist, 0, hist);
    q.writeBuffer(this.pScan1, 0, new Uint32Array([this.cellCount, 0, 0, 0]));
    q.writeBuffer(this.pScan2, 0, new Uint32Array([this.numBlocks, 0, 0, 0]));
    q.writeBuffer(this.pAdd, 0, new Uint32Array([this.cellCount, cfg.agents, 0, 0]));
    q.writeBuffer(this.pScatter, 0, new Uint32Array([cfg.agents, 0, 0, 0]));

    const pipe = (code: string, label: string) =>
      device.createComputePipeline({
        label,
        layout: "auto",
        compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" },
      });

    this.histPipe = pipe(histogramWgsl, "histogram");
    this.scanPipe = pipe(scanWgsl, "scan");
    this.addPipe = pipe(addOffsetsWgsl, "addOffsets");
    this.scatterPipe = pipe(scatterWgsl, "scatter");

    const bind = (p: GPUComputePipeline, res: GPUBuffer[], label: string) =>
      device.createBindGroup({
        label,
        layout: p.getBindGroupLayout(0),
        entries: res.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

    this.bgHist = bind(this.histPipe, [this.pHist, this.pos, this.cellOf, this.counts], "bg:hist");
    this.bgScan1 = bind(
      this.scanPipe,
      [this.pScan1, this.counts, this.offsets, this.blockSums],
      "bg:scan1",
    );
    this.bgScan2 = bind(
      this.scanPipe,
      [this.pScan2, this.blockSums, this.blockOff, this.blockSums2],
      "bg:scan2",
    );
    this.bgAdd = bind(this.addPipe, [this.pAdd, this.offsets, this.blockOff], "bg:add");
    this.bgScatter = bind(
      this.scatterPipe,
      [this.pScatter, this.cellOf, this.cursor, this.sortedIdx],
      "bg:scatter",
    );
  }

  /** Record the full rebin into an existing encoder. */
  record(encoder: GPUCommandEncoder, timer?: PassTimer): void {
    const agentGroups = Math.ceil(this.cfg.agents / WG);
    const cellGroups = Math.ceil((this.cellCount + 1) / WG);

    encoder.clearBuffer(this.counts);

    const run = (
      pipeline: GPUComputePipeline,
      group: GPUBindGroup,
      groups: number,
      label: string,
    ) => {
      const writes = timer?.slot(label);
      const pass = encoder.beginComputePass(writes ? { label, timestampWrites: writes } : { label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
      pass.end();
    };

    run(this.histPipe, this.bgHist, agentGroups, "histogram");
    run(this.scanPipe, this.bgScan1, this.numBlocks, "scan:cells");
    run(this.scanPipe, this.bgScan2, 1, "scan:blocks");
    run(this.addPipe, this.bgAdd, cellGroups, "addOffsets");

    encoder.copyBufferToBuffer(this.offsets, 0, this.cursor, 0, this.cellCount * 4);

    run(this.scatterPipe, this.bgScatter, agentGroups, "scatter");
  }

  /** Upload agent positions as interleaved x,y pairs. */
  writePositions(data: Float32Array): void {
    this.device.queue.writeBuffer(this.pos, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }

  /**
   * Read the whole structure back and check it against the CPU. Slow and only
   * for development - this is the thing that proves the sort is actually a
   * sort, rather than something that merely renders plausibly.
   */
  async validate(): Promise<ValidationReport> {
    const n = this.cfg.agents;
    const cells = this.cellCount;
    const failures: string[] = [];

    const counts = new Uint32Array(await readBuffer(this.device, this.counts, cells * 4));
    const offsets = new Uint32Array(await readBuffer(this.device, this.offsets, (cells + 1) * 4));
    const cellOf = new Uint32Array(await readBuffer(this.device, this.cellOf, n * 4));
    const sorted = new Uint32Array(await readBuffer(this.device, this.sortedIdx, n * 4));

    let total = 0;
    let occupied = 0;
    let maxOcc = 0;
    for (let c = 0; c < cells; c++) {
      const k = counts[c];
      total += k;
      if (k > 0) occupied++;
      if (k > maxOcc) maxOcc = k;
    }
    if (total !== n) failures.push(`counts sum to ${total}, expected ${n}`);

    if (offsets[0] !== 0) failures.push(`offsets[0] is ${offsets[0]}, expected 0`);
    if (offsets[cells] !== n) failures.push(`sentinel offsets[${cells}] is ${offsets[cells]}, expected ${n}`);

    for (let c = 0; c < cells; c++) {
      const span = offsets[c + 1] - offsets[c];
      if (span !== counts[c]) {
        failures.push(`cell ${c}: offset span ${span} != count ${counts[c]}`);
        break;
      }
    }

    const seen = new Uint8Array(n);
    let dupes = 0;
    for (let k = 0; k < n; k++) {
      const a = sorted[k];
      if (a >= n) {
        failures.push(`sortedIdx[${k}] = ${a} is out of range`);
        break;
      }
      if (seen[a]++) dupes++;
    }
    if (dupes > 0) failures.push(`${dupes} agents appear more than once in sortedIdx`);

    let misfiled = 0;
    for (let c = 0; c < cells && misfiled === 0; c++) {
      for (let k = offsets[c]; k < offsets[c + 1]; k++) {
        if (cellOf[sorted[k]] !== c) {
          misfiled++;
          failures.push(`slot ${k} holds agent ${sorted[k]} of cell ${cellOf[sorted[k]]}, in cell ${c}'s range`);
          break;
        }
      }
    }

    return {
      ok: failures.length === 0,
      failures,
      agents: n,
      cells,
      occupiedCells: occupied,
      maxCellOccupancy: maxOcc,
      meanOccupancy: occupied > 0 ? n / occupied : 0,
    };
  }
}
