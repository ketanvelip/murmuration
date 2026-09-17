// One instanced quad per bird, swept backwards along its velocity into a
// tapered streak. Faster birds draw longer, which is what reads as motion in a
// still frame.
//
// Dark ink on a bright sky, alpha-blended: where the flock is dense the strokes
// accumulate toward opaque, and the thin edges stay translucent. That density
// gradient is the murmuration - it is not a lighting effect, it is just overlap.

struct Camera {
  world     : vec2f,
  streak    : f32,
  halfWidth : f32,

  ink       : vec4f,
  agitated  : vec4f,

  predator  : vec2f,
  predR2    : f32,
  alpha     : f32,
};

@group(0) @binding(0) var<uniform>       C     : Camera;
@group(0) @binding(1) var<storage, read> pos   : array<vec2f>;
@group(0) @binding(2) var<storage, read> vel   : array<vec2f>;
@group(0) @binding(3) var<storage, read> alive : array<u32>;

struct VOut {
  @builtin(position) clip  : vec4f,
  @location(0)       tint  : vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;

  if (alive[ii] == 0u) {
    // Park dead birds outside the frustum rather than drawing a zero-area quad,
    // which would still cost rasterisation.
    o.clip = vec4f(-10.0, -10.0, 0.0, 1.0);
    o.tint = vec4f(0.0);
    return o;
  }

  var corners = array<vec2f, 6>(
    vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(0.0, -1.0), vec2f(1.0,  1.0), vec2f(0.0, 1.0),
  );

  let p  = pos[ii];
  let v  = vel[ii];
  let sp = length(v);
  let dir = select(vec2f(1.0, 0.0), v / sp, sp > 1e-4);
  let nrm = vec2f(-dir.y, dir.x);

  let c = corners[vi];
  let taper = 1.0 - 0.62 * c.x;
  let world = p - dir * (C.streak * sp) * c.x + nrm * (C.halfWidth * taper) * c.y;

  let ndc = vec2f(
     world.x / C.world.x * 2.0 - 1.0,
     1.0 - world.y / C.world.y * 2.0,
  );
  o.clip = vec4f(ndc, 0.0, 1.0);

  // Birds near the predator take the agitated tint.
  let d = p - C.predator;
  let heat = clamp(1.0 - dot(d, d) / max(C.predR2, 1.0), 0.0, 1.0);
  let col = mix(C.ink.rgb, C.agitated.rgb, heat);
  o.tint = vec4f(col, C.alpha * mix(1.0, 1.25, heat));
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(in.tint.rgb * in.tint.a, in.tint.a);
}
