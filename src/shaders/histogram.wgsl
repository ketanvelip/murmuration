// Pass 1 of the spatial counting sort.
// Assigns every agent to a uniform grid cell and counts cell occupancy.

struct Params {
  n         : u32,
  cellCount : u32,
  gridX     : u32,
  gridY     : u32,
  cellSize  : f32,
  worldX    : f32,
  worldY    : f32,
  _pad      : f32,
};

@group(0) @binding(0) var<uniform>                  P       : Params;
@group(0) @binding(1) var<storage, read>            pos     : array<vec2f>;
@group(0) @binding(2) var<storage, read_write>      cellOf  : array<u32>;
@group(0) @binding(3) var<storage, read_write>      counts  : array<atomic<u32>>;

fn cellIndex(p: vec2f) -> u32 {
  // Clamp before the float->uint cast: u32() of a negative float is undefined
  // in WGSL, and an agent one frame outside the world would poison the bin.
  let q  = clamp(p, vec2f(0.0, 0.0), vec2f(P.worldX - 0.001, P.worldY - 0.001));
  let gx = min(u32(q.x / P.cellSize), P.gridX - 1u);
  let gy = min(u32(q.y / P.cellSize), P.gridY - 1u);
  return gy * P.gridX + gx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cellIndex(pos[i]);
  cellOf[i] = c;
  atomicAdd(&counts[c], 1u);
}
