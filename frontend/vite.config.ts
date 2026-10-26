import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const demo = process.env.VITE_DEMO === "1";

export default defineConfig({
  plugins: [react()],
  // Demo build (GitHub Pages): served under /touchline-ai/, with the recorded run's data instead of an API.
  base: demo ? "/touchline-ai/" : "/",
  publicDir: demo ? "demo-public" : "public",
  build: { outDir: demo ? "dist-demo" : "dist" },
  server: {
    proxy: { "/api": process.env.TOUCHLINE_API ?? "http://localhost:8780" },
  },
});
