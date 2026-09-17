// Gather agent data into cell order, so the steering pass reads neighbours
// contiguously.
//
// Without this, flocking reaches a neighbour as sortedIdx[k] -> posIn[j], and
// that indirection scatters reads across the whole position buffer - every lane
// in a warp pulling from a different cache line. After this pass a cell's
// agents are adjacent in memory, offsets[] indexes straight into the arrays,
// and the indirection disappears from the inner loop entirely.
//
// The reorder itself is the scattered read, but it happens once per agent
// instead of once per neighbour inspected.

struct Params {
  n   : u32,
  _p0 : u32,
  _p1 : u32,
  _p2 : u32,
};

@group(0) @binding(0) var<uniform>             P         : Params;
@group(0) @binding(1) var<storage, read>       sortedIdx : array<u32>;
@group(0) @binding(2) var<storage, read>       posIn     : array<vec2f>;
@group(0) @binding(3) var<storage, read>       velIn     : array<vec2f>;
@group(0) @binding(4) var<storage, read>       aliveIn   : array<u32>;
@group(0) @binding(5) var<storage, read_write> posOut    : array<vec2f>;
@group(0) @binding(6) var<storage, read_write> velOut    : array<vec2f>;
@group(0) @binding(7) var<storage, read_write> aliveOut  : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.n) { return; }
  let s = sortedIdx[k];
  posOut[k]   = posIn[s];
  velOut[k]   = velIn[s];
  aliveOut[k] = aliveIn[s];
}
