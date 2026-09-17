// The dusk gradient, drawn over the whole scene every frame at partial alpha.
//
// That partial alpha is the trail mechanism: the previous frame is not cleared,
// it is washed toward the sky colour, so birds leave a short smear behind them.
// Raising `fade.x` shortens the trails, lowering it lengthens them.

struct SkyU {
  c0   : vec4f,   // zenith
  c1   : vec4f,
  c2   : vec4f,
  c3   : vec4f,   // horizon
  fade : vec4f,   // x = alpha this frame
};

@group(0) @binding(0) var<uniform> S : SkyU;

struct VOut {
  @builtin(position) clip : vec4f,
  @location(0)       t    : f32,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  let v = p[i];
  o.clip = vec4f(v, 0.0, 1.0);
  o.t = 1.0 - (v.y + 1.0) * 0.5;   // 0 at the top of the screen
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let t = clamp(in.t, 0.0, 1.0);
  var col: vec3f;
  if (t < 0.45) {
    col = mix(S.c0.rgb, S.c1.rgb, t / 0.45);
  } else if (t < 0.78) {
    col = mix(S.c1.rgb, S.c2.rgb, (t - 0.45) / 0.33);
  } else {
    col = mix(S.c2.rgb, S.c3.rgb, (t - 0.78) / 0.22);
  }
  let a = S.fade.x;
  return vec4f(col * a, a);
}
