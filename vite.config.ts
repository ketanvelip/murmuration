import { defineConfig } from "vite";

export default defineConfig({
  build: { target: "esnext" },
  server: { port: 5273, strictPort: true },
});
