import { initGpu } from "./gpu/device";
import { PassTimer } from "./gpu/timing";
import { SpatialSort } from "./sim/spatial";
import type { ValidationReport } from "./sim/spatial";

/**
 * Spike harness for the GPU spatial sort.
 *
 * Answers one question before any of the rest of the pipeline gets built:
 * can we rebin N agents per frame on the GPU, correctly, inside a frame budget?
 *
 * Correctness is checked by reading the whole structure back and verifying it
 * on the CPU. Timing uses timestamp-query, so the numbers are real GPU time
 * and do not depend on requestAnimationFrame.
 */

const CELL_SIZE = 8;

interface Case {
  agents: number;
  grid: number;
}

// 1M runs twice, first and last. NVIDIA parts idle at low clocks and take a
// while to boost, so an early case can be measured on a cold GPU. The repeat
// is the control: if the two 1M numbers disagree, the early ones are clock
// ramp rather than anything structural.
const CASES: Case[] = [
  { agents: 250_000, grid: 256 },
  { agents: 1_000_000, grid: 512 },
  { agents: 4_000_000, grid: 1024 },
  { agents: 1_000_000, grid: 512 },
];

// Warm up by wall clock, not by iteration count. NVIDIA parts idle around
// 300 MHz and take a few hundred milliseconds to boost; measuring before that
// reported 7.54 ms for a case whose settled cost is 0.65 ms - an 11x error
// that looks entirely plausible if you don't check it.
const WARMUP_MS = 600;
const WARMUP_MAX_ITERS = 4000;
const ITERS = 40;

interface CaseResult {
  agents: number;
  cells: number;
  validation: ValidationReport;
  passesMs: Record<string, number>;
  totalMs: number;
  occupancy: { occupied: number; max: number; mean: number };
}

const out = document.getElementById("out") as HTMLElement;

function line(text: string, cls = ""): void {
  const el = document.createElement("div");
  if (cls) el.className = cls;
  el.textContent = text;
  out.appendChild(el);
}

function fmt(n: number, places = 3): string {
  return n.toFixed(places);
}

async function runCase(device: GPUDevice, timestamps: boolean, c: Case): Promise<CaseResult> {
  const sort = new SpatialSort(device, {
    agents: c.agents,
    gridX: c.grid,
    gridY: c.grid,
    cellSize: CELL_SIZE,
  });

  // Clustered rather than uniform: real flocks leave most cells empty and pack
  // a few hard, which is the case that stresses the atomics.
  const world = c.grid * CELL_SIZE;
  const pos = new Float32Array(c.agents * 2);
  const clusters = 24;
  for (let i = 0; i < c.agents; i++) {
    const k = i % clusters;
    const cx = ((k * 7919) % 1000) / 1000 * world;
    const cy = ((k * 6271) % 1000) / 1000 * world;
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * Math.random() * world * 0.18;
    pos[i * 2] = cx + Math.cos(a) * r;
    pos[i * 2 + 1] = cy + Math.sin(a) * r;
  }
  sort.writePositions(pos);

  const timer = new PassTimer(device, timestamps);

  const once = async (measure: boolean) => {
    if (measure) timer.begin();
    const enc = device.createCommandEncoder();
    sort.record(enc, measure ? timer : undefined);
    if (measure) timer.finish(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return measure ? timer.results() : Promise.resolve({});
  };

  const warmStart = performance.now();
  let warmIters = 0;
  while (performance.now() - warmStart < WARMUP_MS && warmIters < WARMUP_MAX_ITERS) {
    await once(false);
    warmIters++;
  }

  const validation = await sort.validate();

  const acc: Record<string, number> = {};
  for (let i = 0; i < ITERS; i++) {
    const r = (await once(true)) as Record<string, number>;
    for (const [k, v] of Object.entries(r)) acc[k] = (acc[k] ?? 0) + v;
  }
  const passesMs: Record<string, number> = {};
  let totalMs = 0;
  for (const [k, v] of Object.entries(acc)) {
    passesMs[k] = v / ITERS;
    totalMs += v / ITERS;
  }

  timer.destroy();

  return {
    agents: c.agents,
    cells: sort.cellCount,
    validation,
    passesMs,
    totalMs,
    occupancy: {
      occupied: validation.occupiedCells,
      max: validation.maxCellOccupancy,
      mean: validation.meanOccupancy,
    },
  };
}

async function main(): Promise<void> {
  line("initialising webgpu…");
  const { adapter, device, timestamps } = await initGpu();

  const info = adapter.info ?? ({} as GPUAdapterInfo);
  line(`adapter: ${info.vendor ?? "?"} / ${info.architecture ?? "?"}`);
  line(`timestamp-query: ${timestamps ? "yes" : "NO - timings unavailable"}`);
  line("");

  const results: CaseResult[] = [];

  for (const c of CASES) {
    line(`${c.agents.toLocaleString()} agents, ${c.grid}x${c.grid} grid…`);
    try {
      const r = await runCase(device, timestamps, c);
      results.push(r);

      const v = r.validation;
      line(`  sort valid: ${v.ok ? "PASS" : "FAIL"}`, v.ok ? "ok" : "bad");
      for (const f of v.failures.slice(0, 5)) line(`    ${f}`, "bad");
      line(
        `  occupancy: ${v.occupiedCells.toLocaleString()} cells used, ` +
          `mean ${fmt(v.meanOccupancy, 1)}, max ${v.maxCellOccupancy}`,
      );
      if (timestamps) {
        for (const [k, ms] of Object.entries(r.passesMs)) line(`  ${k.padEnd(14)} ${fmt(ms)} ms`);
        line(`  ${"TOTAL".padEnd(14)} ${fmt(r.totalMs)} ms`, "total");
      }
    } catch (err) {
      line(`  ERROR: ${(err as Error).message}`, "bad");
    }
    line("");
  }

  (window as unknown as { __spike: unknown }).__spike = {
    adapter: { vendor: info.vendor, architecture: info.architecture },
    timestamps,
    results,
  };
  line("done.", "total");
}

main().catch((err: Error) => {
  line(`FATAL: ${err.message}`, "bad");
  (window as unknown as { __spike: unknown }).__spike = { fatal: err.message };
});
