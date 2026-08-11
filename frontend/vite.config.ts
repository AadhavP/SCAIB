import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev proxy runs in Node, so it can hold the API bearer token. Reading it
// from `process.env` keeps it out of the client bundle: a `VITE_`-prefixed
// variable would be inlined into the served JavaScript and become public. The
// browser's EventSource cannot set headers, so proxy-side injection is also the
// only way SSE can authenticate.
const apiKey = process.env.AGENT_EVALS_API__API_KEY

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8004',
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
      },
    },
  },
})
