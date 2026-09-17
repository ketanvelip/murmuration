// Pass 2 of the spatial counting sort: exclusive prefix sum.
//
// Each workgroup scans BLOCK (1024) elements and emits its total to blockSums.
// The host runs this same pipeline twice - once over the cell counts, once over
// the block sums themselves - then addOffsets folds the second result back into
// the first. Two levels covers cellCount up to BLOCK^2 = 1,048,576 cells.
//
// Per thread: a serial scan of PER elements, then a Hillis-Steele scan across
// the 256 thread totals in workgroup memory.

struct Params {
  n    : u32,   // number of elements to scan
  _p0  : u32,
  _p1  : u32,
  _p2  : u32,
};

@group(0) @binding(0) var<uniform>             P         : Params;
@group(0) @binding(1) var<storage, read>       inp       : array<u32>;
@group(0) @binding(2) var<storage, read_write> outp      : array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums : array<u32>;

const WG    : u32 = 256u;
const PER   : u32 = 4u;
const BLOCK : u32 = 1024u;   // WG * PER

var<workgroup> sdata : array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3u,
        @builtin(workgroup_id)        wid : vec3u) {

  let tid  = lid.x;
  let base = wid.x * BLOCK + tid * PER;

  // Serial exclusive scan over this thread's PER elements.
  var local : array<u32, 4>;
  var sum : u32 = 0u;
  for (var i : u32 = 0u; i < PER; i = i + 1u) {
    let idx = base + i;
    var x : u32 = 0u;
    if (idx < P.n) { x = inp[idx]; }
    local[i] = sum;
    sum = sum + x;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  // Hillis-Steele inclusive scan across thread totals.
  for (var off : u32 = 1u; off < WG; off = off << 1u) {
    var t : u32 = 0u;
    if (tid >= off) { t = sdata[tid - off]; }
    workgroupBarrier();
    if (tid >= off) { sdata[tid] = sdata[tid] + t; }
    workgroupBarrier();
  }

  // Convert this thread's inclusive total into its exclusive base.
  var threadOff : u32 = 0u;
  if (tid > 0u) { threadOff = sdata[tid - 1u]; }

  for (var i : u32 = 0u; i < PER; i = i + 1u) {
    let idx = base + i;
    if (idx < P.n) { outp[idx] = local[i] + threadOff; }
  }

  if (tid == WG - 1u) { blockSums[wid.x] = sdata[WG - 1u]; }
}
