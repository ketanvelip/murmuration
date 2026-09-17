// Present the persistent scene texture to the swapchain.
//
// Swapchain textures are transient - their contents are not preserved between
// frames - so the trail accumulation has to live in a texture we own, and this
// pass copies it out each frame.

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var tex  : texture_2d<f32>;

struct VOut {
  @builtin(position) clip : vec4f,
  @location(0)       uv   : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  let v = p[i];
  o.clip = vec4f(v, 0.0, 1.0);
  o.uv = vec2f((v.x + 1.0) * 0.5, 1.0 - (v.y + 1.0) * 0.5);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(textureSample(tex, samp, in.uv).rgb, 1.0);
}
