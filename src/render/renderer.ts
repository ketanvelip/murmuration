import skyWgsl from "../shaders/sky.wgsl?raw";
import birdsWgsl from "../shaders/birds.wgsl?raw";
import blitWgsl from "../shaders/blit.wgsl?raw";

/** Dusk, and the same sky an hour later. Waves interpolate between them. */
const SKY_DUSK = [
  [0.169, 0.208, 0.333],
  [0.357, 0.345, 0.471],
  [0.659, 0.478, 0.447],
  [0.937, 0.753, 0.553],
];
const SKY_NIGHT = [
  [0.043, 0.059, 0.11],
  [0.078, 0.102, 0.169],
  [0.141, 0.141, 0.227],
  [0.263, 0.188, 0.235],
];
const INK_DUSK = [0.043, 0.055, 0.098];
const INK_NIGHT = [0.576, 0.6, 0.706];
const AGITATED_DUSK = [0.361, 0.141, 0.22];
const AGITATED_NIGHT = [0.745, 0.471, 0.549];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerp3 = (a: number[], b: number[], t: number): [number, number, number] => [
  lerp(a[0] as number, b[0] as number, t),
  lerp(a[1] as number, b[1] as number, t),
  lerp(a[2] as number, b[2] as number, t),
];

export interface FrameLook {
  /** 0 = dusk, 1 = night. Drives sky, ink and agitation colour together. */
  dusk: number;
  /** Sky alpha per frame. Lower leaves longer trails. */
  fade: number;
  predator: [number, number];
  predatorRadius: number;
  streak: number;
  halfWidth: number;
  inkAlpha: number;
}

export class Renderer {
  private readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: GPUCanvasContext;
  private readonly format: GPUTextureFormat;

  private readonly skyPipe: GPURenderPipeline;
  private readonly birdPipe: GPURenderPipeline;
  private readonly blitPipe: GPURenderPipeline;

  private readonly skyU: GPUBuffer;
  private readonly birdU: GPUBuffer;
  private readonly skyBG: GPUBindGroup;
  private readonly sampler: GPUSampler;

  private birdBG: GPUBindGroup | null = null;
  private scene: GPUTexture | null = null;
  private blitBG: GPUBindGroup | null = null;
  private needsClear = true;
  private world: [number, number] = [1, 1];

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("Could not acquire a WebGPU canvas context.");
    this.ctx = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: this.format, alphaMode: "opaque" });

    this.skyU = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.birdU = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Premultiplied: both shaders emit colour already multiplied by alpha.
    const blend: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };

    const skyMod = device.createShaderModule({ code: skyWgsl, label: "sky" });
    this.skyPipe = device.createRenderPipeline({
      label: "sky",
      layout: "auto",
      vertex: { module: skyMod, entryPoint: "vs" },
      fragment: { module: skyMod, entryPoint: "fs", targets: [{ format: "rgba8unorm", blend }] },
      primitive: { topology: "triangle-list" },
    });

    const birdMod = device.createShaderModule({ code: birdsWgsl, label: "birds" });
    this.birdPipe = device.createRenderPipeline({
      label: "birds",
      layout: "auto",
      vertex: { module: birdMod, entryPoint: "vs" },
      fragment: { module: birdMod, entryPoint: "fs", targets: [{ format: "rgba8unorm", blend }] },
      primitive: { topology: "triangle-list" },
    });

    const blitMod = device.createShaderModule({ code: blitWgsl, label: "blit" });
    this.blitPipe = device.createRenderPipeline({
      label: "blit",
      layout: "auto",
      vertex: { module: blitMod, entryPoint: "vs" },
      fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    this.skyBG = device.createBindGroup({
      layout: this.skyPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.skyU } }],
    });

    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  }

  /** Bind the flock's agent state. Call once after construction. */
  setBuffers(pos: GPUBuffer, vel: GPUBuffer, alive: GPUBuffer, world: [number, number]): void {
    this.world = world;
    this.birdBG = this.device.createBindGroup({
      label: "bg:birds",
      layout: this.birdPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.birdU } },
        { binding: 1, resource: { buffer: pos } },
        { binding: 2, resource: { buffer: vel } },
        { binding: 3, resource: { buffer: alive } },
      ],
    });
  }

  /** Size the drawing buffer to the element. Returns true if it changed. */
  resize(): boolean {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.scene) return false;

    this.canvas.width = w;
    this.canvas.height = h;
    this.scene?.destroy();
    this.scene = this.device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "scene",
    });
    this.blitBG = this.device.createBindGroup({
      layout: this.blitPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.scene.createView() },
      ],
    });
    this.needsClear = true;
    return true;
  }

  render(encoder: GPUCommandEncoder, agents: number, look: FrameLook): void {
    if (!this.scene || !this.blitBG) return;
    const t = Math.max(0, Math.min(1, look.dusk));

    const sky = new Float32Array(20);
    for (let i = 0; i < 4; i++) {
      const c = lerp3(SKY_DUSK[i] as number[], SKY_NIGHT[i] as number[], t);
      sky.set([c[0], c[1], c[2], 1], i * 4);
    }
    sky.set([look.fade, 0, 0, 0], 16);
    this.device.queue.writeBuffer(this.skyU, 0, sky);

    const ink = lerp3(INK_DUSK, INK_NIGHT, t);
    const hot = lerp3(AGITATED_DUSK, AGITATED_NIGHT, t);
    const bird = new Float32Array(16);
    bird.set([this.world[0], this.world[1], look.streak, look.halfWidth], 0);
    bird.set([ink[0], ink[1], ink[2], 1], 4);
    bird.set([hot[0], hot[1], hot[2], 1], 8);
    bird.set(
      [look.predator[0], look.predator[1], look.predatorRadius * look.predatorRadius, look.inkAlpha],
      12,
    );
    this.device.queue.writeBuffer(this.birdU, 0, bird);

    const scenePass = encoder.beginRenderPass({
      label: "scene",
      colorAttachments: [
        {
          view: this.scene.createView(),
          loadOp: this.needsClear ? "clear" : "load",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    this.needsClear = false;

    scenePass.setPipeline(this.skyPipe);
    scenePass.setBindGroup(0, this.skyBG);
    scenePass.draw(3);

    const bg = this.birdBG;
    if (bg) {
      scenePass.setPipeline(this.birdPipe);
      scenePass.setBindGroup(0, bg);
      scenePass.draw(6, agents);
    }
    scenePass.end();

    const present = encoder.beginRenderPass({
      label: "present",
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    present.setPipeline(this.blitPipe);
    present.setBindGroup(0, this.blitBG);
    present.draw(3);
    present.end();
  }
}
