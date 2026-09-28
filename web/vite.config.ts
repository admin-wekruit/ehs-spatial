import { defineConfig } from "vite";
export default defineConfig({
  base: "./",
  build: { rollupOptions: { input: "app.html" } },
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8792", changeOrigin: true }, "/fast": { target: "http://127.0.0.1:8793" } },
  },
});
