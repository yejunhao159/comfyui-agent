import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    lib: {
      entry: resolve(__dirname, "src/main.tsx"),
      formats: ["es"],
      fileName: () => "main.js",
    },
    rollupOptions: {
      // Don't externalize anything — bundle everything into main.js
      output: {
        // Single chunk for simplicity
        manualChunks: undefined,
        assetFileNames: "[name][extname]",
      },
    },
    cssCodeSplit: false,
    // Inline small assets
    assetsInlineLimit: 8192,
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
});
