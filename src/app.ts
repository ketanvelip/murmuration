import { initGpu } from "./gpu/device";
import { PassTimer } from "./gpu/timing";
import { Flock } from "./sim/flock";
import { Renderer, CRAFT_STRIDE, MAX_CRAFT } from "./render/renderer";
import { Collider, PROBE_STRIDE, STATUS_KILLS, STATUS_PLAYER_HITS, STATUS_SIZE } from "./sim/collide";
import { Readback } from "./gpu/readback";

const CELL = 8;
/** Mean birds per cell. Drives how fine the grid is for a given flock size. */
const TARGET_OCCUPANCY = 5.2;

const MAX_PULSES = 96;
const PULSE_SPEED = 620;
const PULSE_LIFE = 1.15;
const PULSE_RADIUS = 5;
const FIRE_INTERVAL = 0.05;
const HULL_RADIUS = 11;

// The drone has mass. It lags the pointer, which is what makes being mobbed a
// problem you have to fly out of rather than teleport away from.
const DRONE_ACCEL = 2100;
const DRONE_MAX_SPEED = 430;
const DRONE_DRAG = 0.0016;

const HIT_DAMAGE = 9;
const HIT_INVULN = 0.7;

const canvas = document.getElementById("view") as HTMLCanvasElement;
const fpsEl = document.getElementById("fps") as HTMLElement;
const gpuEl = document.getElementById("gpums") as HTMLElement;
const countEl = document.getElementById("count") as HTMLElement;
const dispersedEl = document.getElementById("dispersed") as HTMLElement;
const integrityEl = document.getElementById("integrity") as HTMLElement;
const barEl = document.getElementById("bar") as HTMLElement;
const overlayEl = document.getElementById("overlay") as HTMLElement;
const overTallyEl = document.getElementById("overtally") as HTMLElement;
const againEl = document.getElementById("again") as HTMLButtonElement;
const select = document.getElementById("agents") as HTMLSelectElement;
const fail = document.getElementById("fail") as HTMLElement;

/** Grid shaped to the canvas so the world is not distorted when drawn. */
function gridFor(agents: number, aspect: number): { gx: number; gy: number } {
  const cells = Math.max(1024, Math.round(agents / TARGET_OCCUPANCY));
  const gy = Math.max(16, Math.round(Math.sqrt(cells / aspect)));
  const gx = Math.max(16, Math.round(gy * aspect));
  return { gx, gy };
}

interface Pulse {
  x: number;
  y: number;
  vx: number;
  vy: number;
  t: number;
}

async function main(): Promise<void> {
  const { device, timestamps } = await initGpu();
  const renderer = new Renderer(device, canvas);
  renderer.resize();

  const aspect = Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight));

  let flock: Flock | null = null;
  let collider: Collider | null = null;
  let readback: Readback | null = null;

  let pulses: Pulse[] = [];
  const probeData = new Float32Array(((MAX_PULSES + 1) * PROBE_STRIDE) / 4);
  const craftData = new Float32Array(MAX_CRAFT * CRAFT_STRIDE);
  let dispersed = 0;

  // Drone
  let droneX = 0;
  let droneY = 0;
  let droneVX = 0;
  let droneVY = 0;
  let integrity = 100;
  let invuln = 0;
  let over = false;

  // Pointer
  let aimAtX = 0;
  let aimAtY = 0;
  let pointerActive = false;
  let firing = false;
  let fireCd = 0;
  let aimX = 1;
  let aimY = 0;
  let spray = 0;

  function reset(): void {
    if (!flock) return;
    droneX = flock.world[0] * 0.5;
    droneY = flock.world[1] * 0.72;
    droneVX = 0;
    droneVY = 0;
    aimAtX = droneX;
    aimAtY = droneY;
    integrity = 100;
    invuln = 1.2;
    over = false;
    dispersed = 0;
    pulses = [];
    dispersedEl.textContent = "0";
    integrityEl.textContent = "100";
    barEl.style.width = "100%";
    barEl.classList.remove("low");
    overlayEl.hidden = true;
  }

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
    reset();
  };

  build(Number(select.value));

  select.addEventListener("change", () => build(Number(select.value)));
  againEl.addEventListener("click", () => reset());

  const toWorld = (ev: PointerEvent) => {
    if (!flock) return;
    const r = canvas.getBoundingClientRect();
    aimAtX = ((ev.clientX - r.left) / r.width) * flock.world[0];
    aimAtY = ((ev.clientY - r.top) / r.height) * flock.world[1];
    pointerActive = true;
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
    pointerActive = false;
    firing = false;
  });

  const timer = new PassTimer(device, timestamps, 14);
  let timingBusy = false;

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

    // Drone: thrust toward the pointer, with drag and a speed cap.
    if (pointerActive && !over) {
      const dx = aimAtX - droneX;
      const dy = aimAtY - droneY;
      const l = Math.hypot(dx, dy);
      if (l > 2) {
        droneVX += (dx / l) * DRONE_ACCEL * dt;
        droneVY += (dy / l) * DRONE_ACCEL * dt;
      }
    }
    const drag = Math.pow(DRONE_DRAG, dt);
    droneVX *= drag;
    droneVY *= drag;
    const sp = Math.hypot(droneVX, droneVY);
    if (sp > DRONE_MAX_SPEED) {
      droneVX = (droneVX / sp) * DRONE_MAX_SPEED;
      droneVY = (droneVY / sp) * DRONE_MAX_SPEED;
    }
    droneX += droneVX * dt;
    droneY += droneVY * dt;
    if (droneX < 10) {
      droneX = 10;
      droneVX = Math.abs(droneVX) * 0.4;
    } else if (droneX > wx - 10) {
      droneX = wx - 10;
      droneVX = -Math.abs(droneVX) * 0.4;
    }
    if (droneY < 10) {
      droneY = 10;
      droneVY = Math.abs(droneVY) * 0.4;
    } else if (droneY > wy - 10) {
      droneY = wy - 10;
      droneVY = -Math.abs(droneVY) * 0.4;
    }
    if (sp > 12) {
      aimX = droneVX / sp;
      aimY = droneVY / sp;
    }

    if (invuln > 0) invuln -= dt;

    const roostX = wx * 0.5 + Math.sin(t * 0.21) * wx * 0.24 + Math.sin(t * 0.09 + 1.3) * wx * 0.1;
    const roostY = wy * 0.46 + Math.cos(t * 0.17) * wy * 0.22 + Math.sin(t * 0.13 + 0.7) * wy * 0.08;

    flock.setDynamics({
      dt,
      roost: [roostX, roostY],
      predator: [droneX, droneY],
      predatorWeight: over ? 0 : 1.35,
      predatorRadius: 260,
      maxSpeed: 150,
      minSpeed: 84,
    });

    // Pulses: spawn, advance, retire.
    fireCd -= dt;
    if (firing && !over && fireCd <= 0 && pulses.length < MAX_PULSES) {
      fireCd = FIRE_INTERVAL;
      // A stationary drone would drill one spot; rotating the spread turns it
      // into a fan the flock has to part around.
      spray += 0.6;
      const jitter = Math.sin(spray) * 0.22;
      const cs = Math.cos(jitter);
      const sn = Math.sin(jitter);
      pulses.push({
        x: droneX,
        y: droneY,
        vx: (aimX * cs - aimY * sn) * PULSE_SPEED + droneVX * 0.3,
        vy: (aimX * sn + aimY * cs) * PULSE_SPEED + droneVY * 0.3,
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
    if (!over) {
      const o = probeCount * 4;
      probeData[o] = droneX;
      probeData[o + 1] = droneY;
      probeData[o + 2] = HULL_RADIUS;
      probeData[o + 3] = 1;
      probeCount++;
    }

    // Visuals for the machine: rounds first, hull last so it sits on top.
    let craftCount = 0;
    for (const u of pulses) {
      if (craftCount >= MAX_CRAFT - 1) break;
      const sp2 = Math.hypot(u.vx, u.vy) || 1;
      const o = craftCount * CRAFT_STRIDE;
      craftData[o] = u.x;
      craftData[o + 1] = u.y;
      craftData[o + 2] = u.vx / sp2;
      craftData[o + 3] = u.vy / sp2;
      craftData[o + 4] = 5.5;
      craftData[o + 5] = 0;
      // Fade over life so a spent round thins out instead of blinking away.
      craftData[o + 6] = 0.9 * (1 - u.t / PULSE_LIFE);
      craftData[o + 7] = 0;
      craftCount++;
    }
    if (!over) {
      const o = craftCount * CRAFT_STRIDE;
      craftData[o] = droneX;
      craftData[o + 1] = droneY;
      craftData[o + 2] = aimX;
      craftData[o + 3] = aimY;
      craftData[o + 4] = 22;
      craftData[o + 5] = 1;
      // Blink while invulnerable, so a hit is legible without a HUD glance.
      craftData[o + 6] = invuln > 0 && Math.floor(t * 18) % 2 === 0 ? 0.35 : 1;
      craftData[o + 7] = 0;
      craftCount++;
    }
    renderer.setCraft(craftData, craftCount);

    const measure = timestamps && !timingBusy && frame % 30 === 0;
    if (measure) {
      timer.begin();
      timingBusy = true;
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
      predator: [droneX, droneY],
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
        // Stop counting once down, or rounds still in flight keep scoring past
        // the tally already shown on the card.
        const k = over ? 0 : latest[STATUS_KILLS] ?? 0;
        if (k > 0) {
          dispersed += k;
          dispersedEl.textContent = dispersed.toLocaleString("en-US");
        }
        // One frame stale, which is invisible on a health bar.
        const hits = latest[STATUS_PLAYER_HITS] ?? 0;
        if (hits > 0 && invuln <= 0 && !over) {
          integrity = Math.max(0, integrity - HIT_DAMAGE);
          invuln = HIT_INVULN;
          integrityEl.textContent = String(Math.round(integrity));
          barEl.style.width = `${integrity}%`;
          barEl.classList.toggle("low", integrity <= 36);
          if (integrity <= 0) {
            over = true;
            firing = false;
            overTallyEl.textContent = dispersed.toLocaleString("en-US");
            overlayEl.hidden = false;
          }
        }
      }
    }

    if (measure) {
      // Resolved off the critical path: awaiting here would stall the pipeline.
      void timer
        .results()
        .then((r) => {
          gpuEl.textContent = Object.values(r)
            .reduce((a, b) => a + b, 0)
            .toFixed(2);
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
