import { defineConfig } from "vite";
export default defineConfig({
  base: "./",
  cacheDir: process.env.VITE_CACHE_DIR,  // a scratch cache when several worktrees share one node_modules
  build: { rollupOptions: { input: "app.html" } },
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8792", changeOrigin: true }, "/fast": { target: `http://127.0.0.1:${process.env.FAST_PORT || 8793}` } },
  },
});
