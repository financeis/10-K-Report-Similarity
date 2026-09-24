import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 빌드 결과는 FastAPI가 서빙한다 (tenksim serve). 개발 중에는 npm run dev + tenksim serve --no-browser.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/tenksim/app/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
