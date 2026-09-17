import { initGpu } from "./gpu/device";
import { PassTimer } from "./gpu/timing";
import { Flock } from "./sim/flock";
import { Renderer } from "./render/renderer";
import { Collider, PROBE_STRIDE, STATUS_KILLS, STATUS_SIZE } from "./sim/collide";
import { Readback } from "./gpu/readback";

const CELL = 8;
const MAX_PULSES = 96;
const PULSE_SPEED = 620;
const PULSE_LIFE = 1.15;
const PULSE_RADIUS = 5;
const FIRE_INTERVAL = 0.05;
const HULL_RADIUS = 11;
/** Mean birds per cell. Drives how fine the grid is for a given flock size. */
const TARGET_OCCUPANCY = 5.2;

const canvas = document.getElementById("view") as HTMLCanvasElement;
const fpsEl = document.getElementById("fps") as HTMLElement;
const gpuEl = document.getElementById("gpums") as HTMLElement;
const countEl = document.getElementById("count") as HTMLElement;
const select = document.getElementById("agents") as HTMLSelectElement;
const dispersedEl = document.getElementById("dispersed") as HTMLElement;
const fail = document.getElementById("fail") as HTMLElement;

/** Grid shaped to the canvas so the world is not distorted when drawn. */
function gridFor(agents: number, aspect: number): { gx: number; gy: number } {
  const cells = Math.max(1024, Math.round(agents / TARGET_OCCUPANCY));
  const gy = Math.max(16, Math.round(Math.sqrt(cells / aspect)));
  const gx = Math.max(16, Math.round(gy * aspect));
  return { gx, gy };
}

async function main(): Promise<void> {
  const { device, timestamps } = await initGpu();
  const renderer = new Renderer(device, canvas);
  renderer.resize();

  const aspect = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));

  let flock: Flock | null = null;
  let collider: Collider | null = null;
  let readback: Readback | null = null;
  let agents = Number(select.value);

  interface Pulse { x: number; y: number; vx: number; vy: number; t: number }
  let pulses: Pulse[] = [];
  const probeData = new Float32Array(((MAX_PULSES + 1) * PROBE_STRIDE) / 4);
  let dispersed = 0;

  const build = (n: number) => {
    const { gx, gy } = gridFor(n, aspect);
    flock = new Flock(device, { agents: n, gridX: gx, gridY: gy, cellSize: CELL });
    collider?.destroy();
    collider = new Collider(device, flock.spatial, MAX_PULSES + 1, {
      pos: flock.pos,
      alive: flock.alive,
    });
    readback?.destroy();
    readback = new Readback(device, STATUS_SIZE, 3);
    renderer.setBuffers(flock.pos, flock.vel, flock.alive, flock.world);
    countEl.textContent = n.toLocaleString("en-US");
    pulses = [];
    dispersed = 0;
    dispersedEl.textContent = "0";
  };
  build(agents);

  select.addEventListener("change", () => {
    agents = Number(select.value);
    build(agents);
  });

  // Predator follows the pointer, in world units.
  let predX = -1e6;
  let predY = -1e6;
  let predActive = false;
  let firing = false;
  let fireCd = 0;
  // Aim follows pointer travel, so rounds go where the drone is heading.
  let aimX = 1;
  let aimY = 0;
  let spray = 0;

  const toWorld = (ev: PointerEvent) => {
    if (!flock) return;
    const r = canvas.getBoundingClientRect();
    const nx = ((ev.clientX - r.left) / r.width) * flock.world[0];
    const ny = ((ev.clientY - r.top) / r.height) * flock.world[1];
    if (predActive) {
      const dx = nx - predX;
      const dy = ny - predY;
      const l = Math.hypot(dx, dy);
      if (l > 1.5) {
        aimX = dx / l;
        aimY = dy / l;
      }
    }
    predX = nx;
    predY = ny;
    predActive = true;
  };
  canvas.addEventListener("pointermove", toWorld);
  canvas.addEventListener("pointerdown", (ev) => {
    canvas.setPointerCapture?.(ev.pointerId);
    toWorld(ev);
    firing = true;
  });
  window.addEventListener("pointerup", () => {
    firing = false;
  });
  canvas.addEventListener("pointerleave", () => {
    predActive = false;
    firing = false;
  });

  const timer = new PassTimer(device, timestamps, 12);
  let timingBusy = false;
  let gpuMs = 0;

  let last = 0;
  let t = 0;
  let frame = 0;
  let fpsAcc = 0;
  let fpsFrames = 0;

  function step(dt: number): void {
    if (!flock) return;
    t += dt;
    frame++;
    renderer.resize();

    const [wx, wy] = flock.world;
    const roostX = wx * 0.5 + Math.sin(t * 0.21) * wx * 0.24 + Math.sin(t * 0.09 + 1.3) * wx * 0.1;
    const roostY = wy * 0.46 + Math.cos(t * 0.17) * wy * 0.22 + Math.sin(t * 0.13 + 0.7) * wy * 0.08;

    flock.setDynamics({
      dt,
      roost: [roostX, roostY],
      predator: [predX, predY],
      predatorWeight: predActive ? 1.35 : 0,
      predatorRadius: 260,
      maxSpeed: 150,
      minSpeed: 84,
    });

    const measure = timestamps && !timingBusy && frame % 30 === 0;
    if (measure) {
      timer.begin();
      timingBusy = true;
    }

    // Pulses: spawn, advance, retire.
    fireCd -= dt;
    if (firing && predActive && fireCd <= 0 && pulses.length < MAX_PULSES) {
      fireCd = FIRE_INTERVAL;
      // A stationary pointer would fire a single stream into one spot; rotating
      // the spread turns it into a fan the flock has to part around.
      spray += 0.6;
      const jitter = Math.sin(spray) * 0.22;
      const cs = Math.cos(jitter);
      const sn = Math.sin(jitter);
      pulses.push({
        x: predX,
        y: predY,
        vx: (aimX * cs - aimY * sn) * PULSE_SPEED,
        vy: (aimX * sn + aimY * cs) * PULSE_SPEED,
        t: 0,
      });
    }
    for (let i = pulses.length - 1; i >= 0; i--) {
      const u = pulses[i] as Pulse;
      u.x += u.vx * dt;
      u.y += u.vy * dt;
      u.t += dt;
      if (u.t > PULSE_LIFE || u.x < -40 || u.x > wx + 40 || u.y < -40 || u.y > wy + 40) {
        pulses.splice(i, 1);
      }
    }

    let probeCount = 0;
    for (const u of pulses) {
      const o = probeCount * 4;
      probeData[o] = u.x;
      probeData[o + 1] = u.y;
      probeData[o + 2] = PULSE_RADIUS;
      probeData[o + 3] = 0;
      probeCount++;
    }
    if (predActive) {
      const o = probeCount * 4;
      probeData[o] = predX;
      probeData[o + 1] = predY;
      probeData[o + 2] = HULL_RADIUS;
      probeData[o + 3] = 1;
      probeCount++;
    }

    const enc = device.createCommandEncoder();

    // Order matters: the index must describe where the birds are when collision
    // tests them, and recordSteer is what moves them.
    flock.recordIndex(enc, measure ? timer : undefined);
    if (collider && probeCount > 0) {
      collider.writeProbes(probeData, probeCount);
      collider.resetStatus(frame);
      collider.record(enc, probeCount, measure ? timer : undefined);
    }
    flock.recordSteer(enc, measure ? timer : undefined);
    if (measure) timer.finish(enc);
    if (collider && readback) readback.record(enc, collider.status);
    renderer.render(enc, flock.cfg.agents, {
      dusk: 0.15,
      fade: 0.3,
      predator: [predActive ? predX : -1e6, predActive ? predY : -1e6],
      predatorRadius: 170,
      streak: 0.055,
      halfWidth: 1.1,
      inkAlpha: 0.82,
    });
    device.queue.submit([enc.finish()]);

    if (readback) {
      readback.kick();
      const latest = readback.latest();
      if (latest) {
        const k = latest[STATUS_KILLS] ?? 0;
        if (k > 0) {
          dispersed += k;
          dispersedEl.textContent = dispersed.toLocaleString("en-US");
        }
      }
    }

    if (measure) {
      // Resolved off the critical path: awaiting here would stall the pipeline.
      void timer
        .results()
        .then((r) => {
          gpuMs = Object.values(r).reduce((a, b) => a + b, 0);
          gpuEl.textContent = gpuMs.toFixed(2);
        })
        .catch(() => {
          // A rejected map must not wedge the flag, or the readout freezes on
          // whatever it happened to be showing and silently stops measuring.
        })
        .finally(() => {
          timingBusy = false;
        });
    }
  }

  function loop(now: number): void {
    requestAnimationFrame(loop);
    if (!last) {
      last = now;
      return;
    }
    let dt = (now - last) / 1000;
    last = now;
    if (dt > 1 / 30) dt = 1 / 30;

    fpsAcc += dt;
    fpsFrames++;
    if (fpsAcc >= 0.5) {
      fpsEl.textContent = (fpsFrames / fpsAcc).toFixed(0);
      fpsAcc = 0;
      fpsFrames = 0;
    }
    step(dt);
  }

  // Dev affordance. A backgrounded tab throttles rAF to nothing, so frames can
  // be driven by hand to check rendering without a foreground window.
  (window as unknown as { __step: (n?: number) => void }).__step = (n = 1) => {
    for (let i = 0; i < n; i++) step(1 / 60);
  };

  requestAnimationFrame(loop);
}

main().catch((err: Error) => {
  fail.hidden = false;
  fail.textContent = err.message;
  console.error(err);
});
