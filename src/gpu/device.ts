export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  timestamps: boolean;
}

/**
 * Acquire a WebGPU device, raising the storage-buffer limits to whatever the
 * adapter will give us. The defaults cap storage bindings at 128 MB, which a
 * few million agents would run into.
 */
export async function initGpu(): Promise<Gpu> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable. Needs Chrome 113+, Edge, or Safari 18+.");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No WebGPU adapter was returned - the GPU may be blocklisted by the browser.");
  }

  const timestamps = adapter.features.has("timestamp-query");

  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
  });

  device.lost.then((info) => {
    console.error(`[gpu] device lost (${info.reason}): ${info.message}`);
  });

  device.addEventListener("uncapturederror", (ev) => {
    console.error("[gpu] uncaptured error:", (ev as GPUUncapturedErrorEvent).error.message);
  });

  return { adapter, device, timestamps };
}

/** Copy a GPU buffer back to the CPU. Allocates a staging buffer per call. */
export async function readBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  size: number,
): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}
