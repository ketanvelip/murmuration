// Pass 4 of the spatial counting sort.
//
// Writes each agent's index into its cell's slot. cursor starts as a copy of
// offsets; the atomic bump hands out consecutive slots within a cell.
//
// Ordering within a cell is nondeterministic - agents race for slots. That is
// fine for neighbour queries, which are order-independent, but it means the
// simulation is not bit-reproducible across runs. Worth knowing before anyone
// tries to build lockstep netcode on this.

struct Params {
  n   : u32,
  _p0 : u32,
  _p1 : u32,
  _p2 : u32,
};

@group(0) @binding(0) var<uniform>             P         : Params;
@group(0) @binding(1) var<storage, read>       cellOf    : array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor    : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIdx : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let c = cellOf[i];
  let k = atomicAdd(&cursor[c], 1u);
  sortedIdx[k] = i;
}
