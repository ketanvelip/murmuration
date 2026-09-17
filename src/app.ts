import { initGpu } from "./gpu/device";
import { PassTimer } from "./gpu/timing";
import { Flock } from "./sim/flock";
import { Renderer } from "./render/renderer";

const CELL = 8;
/** Mean birds per cell. Drives how fine the grid is for a given flock size. */
const TARGET_OCCUPANCY = 5.2;

const canvas = document.getElementById("view") as HTMLCanvasElement;
const fpsEl = document.getElementById("fps") as HTMLElement;
const gpuEl = document.getElementById("gpums") as HTMLElement;
const countEl = document.getElementById("count") as HTMLElement;
const select = document.getElementById("agents") as HTMLSelectElement;
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
  let agents = Number(select.value);

  const build = (n: number) => {
    const { gx, gy } = gridFor(n, aspect);
    flock = new Flock(device, { agents: n, gridX: gx, gridY: gy, cellSize: CELL });
    renderer.setBuffers(flock.pos, flock.vel, flock.alive, flock.world);
    countEl.textContent = n.toLocaleString("en-US");
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
  const toWorld = (ev: PointerEvent) => {
    if (!flock) return;
    const r = canvas.getBoundingClientRect();
    predX = ((ev.clientX - r.left) / r.width) * flock.world[0];
    predY = ((ev.clientY - r.top) / r.height) * flock.world[1];
    predActive = true;
  };
  canvas.addEventListener("pointermove", toWorld);
  canvas.addEventListener("pointerdown", toWorld);
  canvas.addEventListener("pointerleave", () => {
    predActive = false;
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

    const enc = device.createCommandEncoder();
    flock.record(enc, measure ? timer : undefined);
    if (measure) timer.finish(enc);
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
