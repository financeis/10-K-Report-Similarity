import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 빌드 결과는 FastAPI가 서빙한다 (tenksim serve). 개발 중에는 npm run dev + tenksim serve --no-browser.
export default defineConfig({
  plugins: [react()],
  // 관계도(Cytoscape.js)가 들어가 한 파일이 500kB를 넘는다. 로컬 앱이라 나누지 않는다.
  build: { outDir: "../src/tenksim/app/static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
