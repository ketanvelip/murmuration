# Murmuration

A murmuration you have to survive. Single-file Canvas 2D game — no build step, no dependencies.

Starlings flock by three local rules and watch only their nearest handful of neighbours. Nobody leads. Murmurations are also partly *mobbing behaviour* — a defence real starlings run against peregrines and sparrowhawks, where a predator inside the flock gets swarmed rather than evaded. You are flying a survey drone that reads as a predator.

![The flock swirling around its roost, drone below](screenshots/flock.jpg)

## Status

**Prototype.** It runs and it's playable, but it is not a finished game.

There is no level structure — a wave counter ticks every 38 seconds and raises flock size, speed and aggression while the sky darkens. There is no win condition; you can only lose. There is no audio at all.

### Correction to an earlier commit

An earlier version of this README reported a bug: frozen HUD, no projectiles, telemetry identical across captures 41 seconds apart. **That diagnosis was wrong and has been retracted.**

The captures were taken from a background browser tab. Chrome throttles `requestAnimationFrame` to near-zero when `document.hidden` is true, and the loop clamps `dt` to 50ms, so the game advanced roughly a dozen frames across those 41 seconds while the screenshot path force-painted each frame on demand. Measured directly: `visibilityState: "hidden"`, `hasFocus: false`, and a 2-second rAF loop that never completed inside a 45-second timeout.

Nothing was broken. The game only ever ran at full speed when the window was in front.

That did expose one real defect, now fixed: a backgrounded game crawled forward silently instead of stopping. It now pauses on `visibilitychange` and offers an explicit resume, and resets the frame clock on return so no time is skipped.

## Layout

```
legacy/index.html   the Canvas 2D prototype — playable, self-contained
src/                the GPU rewrite — WebGPU compute, TypeScript
screenshots/
```

## Running it

**The GPU spike** needs Node and a WebGPU-capable browser:

```
npm install
npm run dev          # http://localhost:5273
```

**The prototype** needs no toolchain, just HTTP — `file://` won't work because of the font stylesheet:

```
python -m http.server 8731     # from legacy/
```

## Controls

| | |
|---|---|
| Fly | `W A S D` or arrow keys |
| Pulse | hold the mouse, aim with it |
| Flare | `Space` — shoves the whole flock, 6.5s cooldown |
| Touch | drag to fly, pulse is automatic |

## How the simulation works

Classic Reynolds steering over a spatial hash, in plain JavaScript on typed arrays.

Each bird each frame reads its neighbours out of a uniform grid — bucketed by counting sort, so the neighbour query is a 3×3 cell scan rather than a scan of the whole flock — and sums four steering forces:

- **Separation** — push away from anyone too close, falling off as `1/d²`. Strongest of the three.
- **Alignment** — match the average heading of neighbours in range. This is what carries a turn across the flock faster than any single bird flies.
- **Cohesion** — drift toward the neighbours' centroid. Weakest; enough to hold the roost together, not enough to collapse it.
- **Mobbing** — steer at the player. Weight climbs with the wave counter.

Plus a drifting **roost attractor** the flock orbits, which is what produces the large swirling shapes rather than a uniform blob, and a soft boundary force that keeps the flock in frame.

![Field notes explaining the three flocking rules](screenshots/field-notes.jpg)

## Art direction

The obvious treatment for a boids demo is glowing particles on black. This does the opposite, because that's what a murmuration actually looks like: **dark birds against a bright dusk sky**, drawn with accumulating alpha so dense parts of the flock go near-opaque and the edges stay thin. The only artificial light in the frame is your drone, rendered additively in cold cyan against the warm sodium horizon.

Difficulty and art direction are the same variable. Each wave, dusk deepens — the sky gradient lerps toward night and the light reading falls from 346 lx to 76 lx. The bird silhouettes lighten as it goes, so they stay legible against the darkening ground.

## Performance ceiling of the prototype

Roughly **2,400 birds at 60fps**. Canvas 2D means every bird crosses the CPU→GPU boundary as path segments, and the neighbour search runs on one thread. That ceiling is why the GPU track below exists.

## The GPU track

A real murmuration is hundreds of thousands of birds. Reaching that means moving the simulation onto the GPU entirely: agent state in storage buffers, spatial binning in compute shaders, instanced rendering straight off the same buffer with no readback.

The riskiest piece is the spatial rebin, so it was built first, standalone, before committing to the rest.

### Spatial binning — counting sort

Not a bitonic sort. Bitonic is `O(n log²n)` — 253 passes for 4M keys. With atomics available a counting sort is `O(n)` in four dispatches:

```
clearBuffer(counts)     built in, no dispatch
histogram               cell per agent, atomic occupancy counts
scan ×2 + addOffsets    exclusive prefix sum over cells
copyBufferToBuffer      offsets → cursor
scatter                 agent indices into per-cell slots
```

The scan is two-level — a workgroup scans 1024 cells and emits a block total, then one workgroup scans the block totals. That covers up to 1024² = 1,048,576 cells before a third level is needed.

### Measured

NVIDIA Turing, verified correct by reading the whole structure back and checking it on the CPU: every agent present exactly once, every cell range holding only agents of that cell, offsets a true prefix sum of counts.

| agents | histogram | scan | scatter | total | valid |
|-------:|----------:|-----:|--------:|------:|:-----:|
| 250,000 | 0.383 ms | 0.110 ms | 0.251 ms | **0.744 ms** | ✅ |
| 1,000,000 | 0.090 ms | 0.029 ms | 0.557 ms | **0.677 ms** | ✅ |
| 4,000,000 | 1.329 ms | 0.074 ms | 5.005 ms | **6.408 ms** | ✅ |

At 1M agents the rebin costs **4% of a 16.6 ms frame**. The question the spike existed to answer is settled: rebinning is not the bottleneck.

Two things worth recording:

**Warm up by wall clock, not iteration count.** A 3-iteration warmup measured 1M agents at 7.54 ms. The settled cost is 0.65 ms — an **11× error**, entirely GPU clock ramp, and completely plausible-looking if unchecked. The harness now warms for 600 ms and runs 1M twice as a control; the two runs agreeing is what makes the table trustworthy.

**4M scales superlinearly** — 9× the scatter cost for 4× the agents. The working set is ~76 MB against a few MB of L2, and the synthetic distribution puts 50,196 agents in one 8×8 cell, which is far denser than any real flock. Under realistic density this should improve; if it doesn't, the fix is workgroup-local privatised counters to cut atomic contention.

### Still open

Collision when agent state lives on the GPU — bullets against birds, player against birds — is the other half of the spike and is not built. Either a compute pass writing to a small async-mapped readback buffer, accepting 1–2 frames of latency, or a coarse CPU mirror. That decision shapes the rest of the architecture.
