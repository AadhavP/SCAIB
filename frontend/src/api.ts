export type Health = { status: string; version: string }
export type Agent = { id: string; type: string; capabilities: string[]; available: boolean }
export type BenchmarkDetail = { id: string; title: string; description: string; version: string; tags: string[]; datasets: unknown[]; tasks: unknown[]; actions: unknown[]; metrics: unknown[] }
export type EvaluationJob = { job_id: string; benchmark_id: string; agent_id: string; status: string; created_at?: string; result?: Record<string, unknown> | null; error?: string | null }
export type RunRequest = { benchmark_id: string; agent_id: string; model?: string; provider?: string; seed?: number; max_cells?: number; max_steps?: number; config_override?: Record<string, unknown> }
export type RunResponse = RunRequest & { job_id: string; status: string }
export type ApiClient = { health: () => Promise<Health>; benchmarks: () => Promise<string[]>; benchmark: (id: string) => Promise<BenchmarkDetail>; agents: () => Promise<Agent[]>; evaluations: () => Promise<EvaluationJob[]>; evaluation: (id: string) => Promise<EvaluationJob>; run: (request: RunRequest) => Promise<RunResponse> }
type Fetcher = typeof fetch

async function request<T>(fetcher: Fetcher, path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetcher(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) } })
  if (!response.ok) {
    let detail = 'Unknown error'
    try { const payload = await response.json() as { detail?: string }; detail = payload.detail ?? detail } catch { /* empty response */ }
    throw new Error(`API request failed (${response.status}): ${detail}`)
  }
  return response.json() as Promise<T>
}

export function createApiClient(fetcher: Fetcher = fetch): ApiClient {
  return {
    health: () => request<Health>(fetcher, '/v1/health', { method: 'GET' }),
    benchmarks: () => request<string[]>(fetcher, '/v1/benchmarks', { method: 'GET' }),
    agents: () => request<Agent[]>(fetcher, '/v1/agents', { method: 'GET' }),
    benchmark: (id) => request<BenchmarkDetail>(fetcher, `/v1/benchmarks/${encodeURIComponent(id)}`, { method: 'GET' }),
    evaluations: () => request<EvaluationJob[]>(fetcher, '/v1/evaluations', { method: 'GET' }),
    evaluation: (id) => request<EvaluationJob>(fetcher, `/v1/evaluations/${encodeURIComponent(id)}`, { method: 'GET' }),
    run: (payload) => request<RunResponse>(fetcher, '/v1/evaluations/run', { method: 'POST', body: JSON.stringify({ ...payload, config_override: { ...(payload.config_override ?? {}) } }) }),
  }
}
