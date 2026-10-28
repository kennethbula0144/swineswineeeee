import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

function offlinePrecacheManifest() {
  return {
    name: 'offline-precache-manifest',
    generateBundle(_, bundle) {
      const assets = Object.values(bundle)
        .filter((entry) => entry.type === 'asset' || entry.type === 'chunk')
        .map((entry) => `/${entry.fileName}`);
      this.emitFile({
        type: 'asset',
        fileName: 'precache-manifest.json',
        source: JSON.stringify([...new Set(['/', '/index.html', ...assets])]),
      });
    },
  };
}

export default defineConfig({
  envDir: '..',
  plugins: [react(), offlinePrecacheManifest()],
});
