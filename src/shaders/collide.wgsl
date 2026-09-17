// Gameplay collision against GPU-resident agents.
//
// One workgroup per probe (a pulse round, or the player's hull). The 64 threads
// of that workgroup stride over the agents in the cells the probe's radius
// touches, using the ranges the spatial sort produced.
//
// The key architectural point: consequences that the SIMULATION needs - a bird
// dying - happen here, on the GPU, with zero latency, because that is where the
// data lives. Only the scalars the HUD needs travel back to the CPU, and those
// can be a couple of frames stale without anyone noticing.

struct Params {
  gridX      : u32,
  gridY      : u32,
  cellCount  : u32,
  probeCount : u32,
  cellSize   : f32,
  worldX     : f32,
  worldY     : f32,
  _pad       : f32,
};

struct Probe {
  pos    : vec2f,
  radius : f32,
  kind   : f32,   // 0 = pulse (kills), 1 = player hull (contacts), 2 = census
};

struct Status {
  kills      : atomic<u32>,
  playerHits : atomic<u32>,
  frame      : u32,   // stamped by the CPU, read back to measure latency
  census     : atomic<u32>,
};

@group(0) @binding(0) var<uniform>             P         : Params;
@group(0) @binding(1) var<storage, read>       probes    : array<Probe>;
@group(0) @binding(2) var<storage, read>       pos       : array<vec2f>;
@group(0) @binding(3) var<storage, read>       offsets   : array<u32>;
@group(0) @binding(4) var<storage, read>       sortedIdx : array<u32>;
@group(0) @binding(5) var<storage, read_write> alive     : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> status    : Status;

const WG : u32 = 64u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id)        wid : vec3u,
        @builtin(local_invocation_id) lid : vec3u) {

  let bi = wid.x;
  if (bi >= P.probeCount) { return; }

  let pr = probes[bi];
  if (pr.radius <= 0.0) { return; }

  let r2 = pr.radius * pr.radius;
  let bound = vec2f(P.worldX - 0.001, P.worldY - 0.001);
  let lo = clamp(pr.pos - vec2f(pr.radius), vec2f(0.0), bound);
  let hi = clamp(pr.pos + vec2f(pr.radius), vec2f(0.0), bound);

  let gx0 = min(u32(lo.x / P.cellSize), P.gridX - 1u);
  let gx1 = min(u32(hi.x / P.cellSize), P.gridX - 1u);
  let gy0 = min(u32(lo.y / P.cellSize), P.gridY - 1u);
  let gy1 = min(u32(hi.y / P.cellSize), P.gridY - 1u);

  for (var gy = gy0; gy <= gy1; gy = gy + 1u) {
    for (var gx = gx0; gx <= gx1; gx = gx + 1u) {
      let c = gy * P.gridX + gx;
      let s = offsets[c];
      let e = offsets[c + 1u];

      for (var k = s + lid.x; k < e; k = k + WG) {
        let a = sortedIdx[k];
        if (atomicLoad(&alive[a]) == 0u) { continue; }

        let d = pos[a] - pr.pos;
        if (dot(d, d) > r2) { continue; }

        if (pr.kind > 1.5) {
          // Census: count birds inside a region without touching them. This is
          // what lets an objective be about the shape of the flock - drive it
          // somewhere, keep it off something - rather than about a body count.
          atomicAdd(&status.census, 1u);
        } else if (pr.kind > 0.5) {
          atomicAdd(&status.playerHits, 1u);
        } else {
          // Exchange rather than store: two rounds overlapping the same bird in
          // one frame must score once, not twice.
          if (atomicExchange(&alive[a], 0u) == 1u) {
            atomicAdd(&status.kills, 1u);
          }
        }
      }
    }
  }
}
