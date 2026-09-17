// The drone and its pulse rounds - the only artificial light in the frame.
//
// Everything else on screen is a bird rendered as dark ink against dusk. These
// are additive, so they read as emitted light rather than silhouette, which is
// what separates the machine from the flock without needing any other cue.
//
// One instanced quad per item, billboarded and oriented along its heading.

struct Camera {
  world  : vec2f,
  _pad0  : f32,
  _pad1  : f32,
};

struct Craft {
  pos    : vec2f,   // world units
  dir    : vec2f,   // unit heading
  size   : f32,
  kind   : f32,     // 0 = round, 1 = hull, 2 = flare shell, 3 = objective zone
  alpha  : f32,
  _pad   : f32,
};

@group(0) @binding(0) var<uniform>       C     : Camera;
@group(0) @binding(1) var<storage, read> items : array<Craft>;

struct VOut {
  @builtin(position) clip : vec4f,
  @location(0)       uv   : vec2f,   // -1..1 across the quad
  @location(1)       tint : vec4f,
  @location(2)       kind : f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0,  1.0), vec2f(-1.0, 1.0),
  );

  let it = items[ii];
  var o: VOut;

  o.kind = it.kind;

  if (it.size <= 0.0) {
    o.clip = vec4f(-10.0, -10.0, 0.0, 1.0);
    o.uv = vec2f(0.0, 0.0);
    o.tint = vec4f(0.0);
    return o;
  }

  let c = corners[vi];
  let dir = it.dir;
  let nrm = vec2f(-dir.y, dir.x);

  // Rounds stretch along their heading into a short dart; the hull stays round
  // so its glow reads the same whichever way you are flying.
  let along = select(3.2, 1.0, it.kind > 0.5);   // rounds stretch, hull and ring stay round
  let world = it.pos + dir * (c.x * it.size * along) + nrm * (c.y * it.size);

  let ndc = vec2f(
    world.x / C.world.x * 2.0 - 1.0,
    1.0 - world.y / C.world.y * 2.0,
  );
  o.clip = vec4f(ndc, 0.0, 1.0);
  o.uv = c;

  var col: vec3f;
  if (it.kind > 2.5) {
    col = vec3f(1.0, 0.56, 0.32);        // objective zone: warm, not machine cyan
  } else if (it.kind > 1.5) {
    col = vec3f(0.62, 0.92, 1.0);        // flare shell
  } else if (it.kind > 0.5) {
    col = vec3f(0.89, 0.97, 1.0);        // hull
  } else {
    col = vec3f(0.55, 0.88, 0.96);       // round
  }
  o.tint = vec4f(col, it.alpha);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let r = length(in.uv);
  var g: f32;
  if (in.kind > 2.5) {
    // Objective zone: a standing boundary, thinner and steadier than the flare
    // so it does not read as something that just happened.
    let e = (r - 0.92) * 26.0;
    g = exp(-e * e);
  } else if (in.kind > 1.5) {
    // Flare: a thin expanding shell, so it reads as a shockwave passing through
    // the flock rather than a light being switched on.
    let e = (r - 0.78) * 7.0;
    g = exp(-e * e);
  } else {
    // Radial falloff, squared for a tighter core and a soft halo.
    let d = clamp(1.0 - r, 0.0, 1.0);
    g = d * d;
  }
  let a = g * in.tint.a;
  return vec4f(in.tint.rgb * a, a);
}
