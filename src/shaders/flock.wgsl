// Reynolds steering over the sorted neighbourhood, one thread per bird.
//
// Reads the cell ranges the counting sort produced, so each bird only looks at
// the 3x3 cells around it rather than the whole flock. Double-buffered: reads
// posIn/velIn, writes posOut/velOut, and the host swaps. Writing in place would
// race, since neighbours read each other's positions within the same dispatch.

struct Params {
  n         : u32,
  gridX     : u32,
  gridY     : u32,
  cellCount : u32,

  cellSize  : f32,
  worldX    : f32,
  worldY    : f32,
  dt        : f32,

  roostX    : f32,
  roostY    : f32,
  maxSpeed  : f32,
  minSpeed  : f32,

  sepW      : f32,
  aliW      : f32,
  cohW      : f32,
  roostW    : f32,

  percept2  : f32,
  sepDist2  : f32,
  maxForce  : f32,
  edge      : f32,

  predX     : f32,
  predY     : f32,
  predW     : f32,
  predR2    : f32,

  frame     : u32,
  _pad0     : u32,
  _pad1     : u32,
  _pad2     : u32,
};

// posIn/velIn are in cell order, produced by the reorder pass. Thread i is
// sorted slot i, so a cell's agents are the contiguous range
// [offsets[c], offsets[c+1]) and neighbour reads are coalesced.
@group(0) @binding(0) var<uniform>             P         : Params;
@group(0) @binding(1) var<storage, read>       posIn     : array<vec2f>;
@group(0) @binding(2) var<storage, read>       velIn     : array<vec2f>;
@group(0) @binding(3) var<storage, read_write> posOut    : array<vec2f>;
@group(0) @binding(4) var<storage, read_write> velOut    : array<vec2f>;
@group(0) @binding(5) var<storage, read>       offsets   : array<u32>;
@group(0) @binding(6) var<storage, read>       aliveIn   : array<u32>;
@group(0) @binding(7) var<storage, read_write> aliveOut  : array<u32>;

// Cap on agents inspected per bird. Cells in a dense roost can hold thousands;
// without this the worst-case thread dominates the whole dispatch. Boids only
// need a handful of neighbours - real starlings track about seven.
const EXAMINE_LIMIT : u32 = 48u;

fn steer(dir: vec2f, v: vec2f, maxSpeed: f32) -> vec2f {
  let l = length(dir);
  if (l < 1e-5) { return vec2f(0.0, 0.0); }
  return dir / l * maxSpeed - v;
}

/** PCG-style integer hash, for respawn scatter. */
fn hash1(x: u32) -> f32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  return f32(h) * 2.3283064e-10;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }

  let bound = vec2f(P.worldX - 0.001, P.worldY - 0.001);

  // Dispersed by a pulse last frame: fly a replacement in off the edge, so the
  // roost stays the same size and the flock reads as continuous rather than
  // slowly eroding.
  if (aliveIn[i] == 0u) {
    // Rejoin in an annulus around the roost rather than entering from the world
    // edge. Edge entry meant a constant stream of birds in transit hazing the
    // borders; rejoining keeps the flock continuous and the action centred.
    let s = i * 9781u + P.frame * 6271u;
    let ang = hash1(s) * 6.2831853;
    let rad = 170.0 + hash1(s + 1u) * 230.0;
    let dir = vec2f(cos(ang), sin(ang));
    let rp = vec2f(P.roostX, P.roostY) + dir * rad;
    // Tangential entry, so replacements sweep into the orbit instead of
    // colliding head-on with the roost.
    let rv = vec2f(-dir.y, dir.x) * (P.maxSpeed * 0.9);

    posOut[i] = clamp(rp, vec2f(4.0, 4.0), bound - vec2f(4.0, 4.0));
    velOut[i] = rv;
    aliveOut[i] = 1u;
    return;
  }
  aliveOut[i] = 1u;

  let p = posIn[i];
  let v = velIn[i];
  let q = clamp(p, vec2f(0.0, 0.0), bound);
  let cx = i32(min(u32(q.x / P.cellSize), P.gridX - 1u));
  let cy = i32(min(u32(q.y / P.cellSize), P.gridY - 1u));

  var sep = vec2f(0.0, 0.0);
  var ali = vec2f(0.0, 0.0);
  var coh = vec2f(0.0, 0.0);
  var count = 0u;
  var examined = 0u;

  for (var oy = -1; oy <= 1; oy = oy + 1) {
    let yy = cy + oy;
    if (yy < 0 || yy >= i32(P.gridY)) { continue; }
    if (examined >= EXAMINE_LIMIT) { break; }

    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let xx = cx + ox;
      if (xx < 0 || xx >= i32(P.gridX)) { continue; }
      if (examined >= EXAMINE_LIMIT) { break; }

      let c = u32(yy) * P.gridX + u32(xx);
      let s = offsets[c];
      let e = offsets[c + 1u];

      for (var k = s; k < e; k = k + 1u) {
        if (examined >= EXAMINE_LIMIT) { break; }
        examined = examined + 1u;

        if (k == i) { continue; }

        let np = posIn[k];
        let d = np - p;
        let d2 = dot(d, d);
        if (d2 > P.percept2 || d2 <= 1e-6) { continue; }

        count = count + 1u;
        ali = ali + velIn[k];
        coh = coh + np;
        if (d2 < P.sepDist2) { sep = sep - d / d2; }
      }
    }
  }

  var f = vec2f(0.0, 0.0);

  if (count > 0u) {
    f = f + steer(ali, v, P.maxSpeed) * P.aliW;
    f = f + steer(coh / f32(count) - p, v, P.maxSpeed) * P.cohW;
  }
  f = f + steer(sep, v, P.maxSpeed) * P.sepW;

  // The roost the flock orbits. This is what produces the large swirling
  // shapes; pure separation/alignment/cohesion gives a drifting blob.
  let toRoost = vec2f(P.roostX, P.roostY) - p;
  let roostD = length(toRoost);
  let roostWeight = select(P.roostW, P.roostW * 3.4, roostD > 240.0);
  f = f + steer(toRoost, v, P.maxSpeed) * roostWeight;

  // Predator: birds mob it rather than scatter, which is what a murmuration
  // actually does to a hawk.
  let toPred = vec2f(P.predX, P.predY) - p;
  let predD2 = dot(toPred, toPred);
  if (P.predW != 0.0 && predD2 < P.predR2 && predD2 > 1e-4) {
    f = f + steer(toPred, v, P.maxSpeed) * P.predW;
  }

  // Soft walls.
  if (q.x < P.edge)              { f.x = f.x + (P.edge - q.x) * 16.0; }
  else if (q.x > P.worldX - P.edge) { f.x = f.x - (q.x - (P.worldX - P.edge)) * 16.0; }
  if (q.y < P.edge)              { f.y = f.y + (P.edge - q.y) * 16.0; }
  else if (q.y > P.worldY - P.edge) { f.y = f.y - (q.y - (P.worldY - P.edge)) * 16.0; }

  let fl = length(f);
  if (fl > P.maxForce) { f = f / fl * P.maxForce; }

  var nv = v + f * P.dt;
  let sp = length(nv);
  if (sp > P.maxSpeed) {
    nv = nv / sp * P.maxSpeed;
  } else if (sp < P.minSpeed) {
    nv = select(vec2f(P.minSpeed, 0.0), nv / sp * P.minSpeed, sp > 1e-5);
  }

  // Reflect at the boundary rather than clamping. Clamping parks a bird exactly
  // on the edge and leaves it there, so escapees pile into a visible line along
  // the border instead of turning back into the roost.
  var np = p + nv * P.dt;
  if (np.x < 0.0) {
    np.x = 0.0;
    nv.x = abs(nv.x);
  } else if (np.x > bound.x) {
    np.x = bound.x;
    nv.x = -abs(nv.x);
  }
  if (np.y < 0.0) {
    np.y = 0.0;
    nv.y = abs(nv.y);
  } else if (np.y > bound.y) {
    np.y = bound.y;
    nv.y = -abs(nv.y);
  }

  velOut[i] = nv;
  posOut[i] = np;
}
