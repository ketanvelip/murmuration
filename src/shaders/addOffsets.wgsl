// Pass 3 of the spatial counting sort.
//
// Folds the scanned block sums back into the per-cell offsets, turning the
// per-block-local exclusive scan into a global one. Also writes the sentinel
// at offsets[cellCount] = n, so a cell's agent range is always
// [offsets[c], offsets[c+1]) with no special case for the final cell.

struct Params {
  n     : u32,   // cellCount
  total : u32,   // agent count, for the sentinel
  _p0   : u32,
  _p1   : u32,
};

@group(0) @binding(0) var<uniform>             P        : Params;
@group(0) @binding(1) var<storage, read_write> offsets  : array<u32>;
@group(0) @binding(2) var<storage, read>       blockOff : array<u32>;

const BLOCK : u32 = 1024u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i > P.n) { return; }
  if (i == P.n) { offsets[i] = P.total; return; }
  offsets[i] = offsets[i] + blockOff[i / BLOCK];
}
