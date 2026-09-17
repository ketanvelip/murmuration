import { initGpu } from "./gpu/device";
import { PassTimer } from "./gpu/timing";
import { Readback } from "./gpu/readback";
import { SpatialSort } from "./sim/spatial";
import type { ValidationReport } from "./sim/spatial";
import { Collider, PROBE_STRIDE, STATUS_FRAME, STATUS_KILLS, STATUS_PLAYER_HITS } from "./sim/collide";

/**
 * A macrotask that background-tab throttling does not clamp.
 *
 * rAF is throttled to near-zero when document.hidden, and setTimeout gets
 * clamped to ~1s. MessageChannel is neither, so the frame loop runs at full
 * speed and mapAsync callbacks still get an event-loop turn to resolve in.
 */
const tickChannel = new MessageChannel();
function nextTick(): Promise<void> {
  return new Promise<void>((resolve) => {
    tickChannel.port1.onmessage = () => resolve();
    tickChannel.port2.postMessage(null);
  });
}

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

interface CollideResult {
  agents: number;
  probes: number;
  frames: number;
  collideMs: number;
  rebinMs: number;
  latency: { min: number; median: number; p95: number; max: number; histogram: Record<number, number> };
  latencyMs: { min: number; median: number; p95: number; max: number };
  samples: number;
  skipped: number;
  kills: number;
  playerContacts: number;
}

/**
 * Second half of the spike: gameplay collision when agent state lives on the
 * GPU, and how stale the numbers the CPU reads back actually are.
 *
 * Deliberately does NOT await queue completion each frame - that would idle the
 * GPU between submits and report a flatteringly low latency. This submits and
 * moves on, the way a real frame loop does.
 */
async function runCollide(
  device: GPUDevice,
  agents: number,
  grid: number,
  frames: number,
  opts: { timings: boolean; ringDepth: number; paceMs?: number },
): Promise<CollideResult> {
  const sort = new SpatialSort(device, { agents, gridX: grid, gridY: grid, cellSize: CELL_SIZE });
  const world = grid * CELL_SIZE;

  const pos = new Float32Array(agents * 2);
  for (let i = 0; i < agents; i++) {
    const k = i % 24;
    const cx = (((k * 7919) % 1000) / 1000) * world;
    const cy = (((k * 6271) % 1000) / 1000) * world;
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * Math.random() * world * 0.18;
    pos[i * 2] = cx + Math.cos(a) * r;
    pos[i * 2 + 1] = cy + Math.sin(a) * r;
  }
  sort.writePositions(pos);

  const collider = new Collider(device, sort);
  const PULSES = 64;
  const probeCount = PULSES + 1; // + the player hull
  const probeData = new Float32Array((probeCount * PROBE_STRIDE) / 4);

  const px = new Float32Array(probeCount);
  const py = new Float32Array(probeCount);
  const vx = new Float32Array(probeCount);
  const vy = new Float32Array(probeCount);
  for (let i = 0; i < probeCount; i++) {
    px[i] = Math.random() * world;
    py[i] = Math.random() * world;
    const a = Math.random() * Math.PI * 2;
    vx[i] = Math.cos(a) * 60;
    vy[i] = Math.sin(a) * 60;
  }

  const readback = new Readback(device, 16, opts.ringDepth);
  const timer = new PassTimer(device, opts.timings);

  const latencies: number[] = [];
  let lastSeenFrame = -1;
  let kills = 0;
  let playerContacts = 0;
  let collideAcc = 0;
  let rebinAcc = 0;
  let timed = 0;

  for (let frame = 0; frame < frames; frame++) {
    const frameStart = performance.now();
    for (let i = 0; i < probeCount; i++) {
      px[i] += vx[i];
      py[i] += vy[i];
      if (px[i] < 0 || px[i] > world) vx[i] = -vx[i];
      if (py[i] < 0 || py[i] > world) vy[i] = -vy[i];
      const o = i * 4;
      probeData[o] = Math.max(0, Math.min(world, px[i]));
      probeData[o + 1] = Math.max(0, Math.min(world, py[i]));
      probeData[o + 2] = i === PULSES ? 10 : 6;
      probeData[o + 3] = i === PULSES ? 1 : 0;
    }
    collider.writeProbes(probeData, probeCount);
    collider.resetStatus(frame);

    // Timing and latency cannot be measured in the same run: reading timestamps
    // back means awaiting the queue, which serialises CPU and GPU and collapses
    // the readback latency being measured to zero.
    const measure = opts.timings && frame >= 40;
    if (measure) timer.begin();

    const enc = device.createCommandEncoder();
    sort.record(enc, measure ? timer : undefined);
    collider.record(enc, probeCount, measure ? timer : undefined);
    if (measure) timer.finish(enc);
    readback.record(enc, collider.status);
    device.queue.submit([enc.finish()]);
    readback.kick();

    if (measure) {
      const r = (await timer.results()) as Record<string, number>;
      const collideMs = r["collide"] ?? 0;
      let rebin = 0;
      for (const [k, v] of Object.entries(r)) if (k !== "collide") rebin += v;
      if (collideMs > 0) {
        collideAcc += collideMs;
        rebinAcc += rebin;
        timed++;
      }
    }

    const latest = readback.latest();
    if (latest) {
      const seen = latest[STATUS_FRAME];
      if (seen !== lastSeenFrame) {
        lastSeenFrame = seen;
        latencies.push(frame - seen);
        kills += latest[STATUS_KILLS];
        playerContacts += latest[STATUS_PLAYER_HITS];
      }
    }

    // Hold a realistic frame cadence. Without this the CPU laps the GPU by
    // hundreds of frames and any latency measured in frames is fiction.
    const pace = opts.paceMs ?? 0;
    do {
      await nextTick();
    } while (pace > 0 && performance.now() - frameStart < pace);
  }

  latencies.sort((a, b) => a - b);
  const histogram: Record<number, number> = {};
  for (const l of latencies) histogram[l] = (histogram[l] ?? 0) + 1;
  const pick = (q: number) => latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * q))] ?? -1;

  const ms = [...readback.latenciesMs].sort((a, b) => a - b);
  const pickMs = (q: number) => {
    const v = ms[Math.min(ms.length - 1, Math.floor(ms.length * q))];
    return v === undefined ? -1 : +v.toFixed(2);
  };

  const result: CollideResult = {
    agents,
    probes: probeCount,
    frames,
    collideMs: timed > 0 ? collideAcc / timed : 0,
    rebinMs: timed > 0 ? rebinAcc / timed : 0,
    latency: {
      min: latencies[0] ?? -1,
      median: pick(0.5),
      p95: pick(0.95),
      max: latencies[latencies.length - 1] ?? -1,
      histogram,
    },
    latencyMs: {
      min: ms.length ? +(ms[0] as number).toFixed(2) : -1,
      median: pickMs(0.5),
      p95: pickMs(0.95),
      max: ms.length ? +(ms[ms.length - 1] as number).toFixed(2) : -1,
    },
    samples: latencies.length,
    skipped: readback.skipped,
    kills,
    playerContacts,
  };

  timer.destroy();
  readback.destroy();
  collider.destroy();
  return result;
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

  line("collision cost, 1,000,000 agents, 65 probes…");
  let timed: CollideResult | null = null;
  let latency: CollideResult | null = null;
  try {
    timed = await runCollide(device, 1_000_000, 512, 300, { timings: true, ringDepth: 3 });
    line(`  collide pass    ${fmt(timed.collideMs)} ms`);
    line(`  rebin           ${fmt(timed.rebinMs)} ms`);
    line(`  frame total     ${fmt(timed.collideMs + timed.rebinMs)} ms`, "total");
    line(`  kills ${timed.kills.toLocaleString()}, player contacts ${timed.playerContacts.toLocaleString()}`);
  } catch (err) {
    line(`  ERROR: ${(err as Error).message}`, "bad");
  }
  line("");

  line("readback latency, loop paced to 60fps, no timestamp sync…");
  try {
    latency = await runCollide(device, 1_000_000, 512, 300, {
      timings: false,
      ringDepth: 4,
      paceMs: 16.6,
    });
    line(
      `  latency  min ${fmt(latency.latencyMs.min, 2)} / median ${fmt(latency.latencyMs.median, 2)} / ` +
        `p95 ${fmt(latency.latencyMs.p95, 2)} / max ${fmt(latency.latencyMs.max, 2)} ms`,
      "ok",
    );
    line(
      `  in frames  min ${latency.latency.min} / median ${latency.latency.median} / ` +
        `p95 ${latency.latency.p95} / max ${latency.latency.max}`,
      "ok",
    );
    line(`  distribution    ${JSON.stringify(latency.latency.histogram)}`);
    line(`  samples ${latency.samples} of ${latency.frames}, ring skips ${latency.skipped}`);
  } catch (err) {
    line(`  ERROR: ${(err as Error).message}`, "bad");
  }
  line("");

  (window as unknown as { __spike: unknown }).__spike = {
    adapter: { vendor: info.vendor, architecture: info.architecture },
    timestamps,
    results,
    collide: timed,
    latency,
  };
  line("done.", "total");
}

main().catch((err: Error) => {
  line(`FATAL: ${err.message}`, "bad");
  (window as unknown as { __spike: unknown }).__spike = { fatal: err.message };
});
