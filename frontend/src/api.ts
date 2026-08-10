export type JsonMap = Record<string, unknown>

export type Health = { status: string; version: string; features?: string[] }
export type Agent = { id: string; type: string; capabilities: string[]; available: boolean }
export type BenchmarkDataset = JsonMap & { id?: string; name?: string; description?: string; organism?: string; modality?: string; expected_observations?: JsonMap }
export type BenchmarkTask = JsonMap & { id?: string; name?: string; objective?: string; description?: string; allowed_actions?: string[]; metrics?: string[] }
export type BenchmarkAction = JsonMap & { id?: string; name?: string; purpose?: string; estimated_cost?: JsonMap }
export type BenchmarkMetric = JsonMap & { id?: string; metric_id?: string; name?: string; description?: string; direction?: string; contributes_to_reward?: boolean }
export type BenchmarkDetail = {
  id: string
  title: string
  description: string
  version: string
  tags: string[]
  datasets: BenchmarkDataset[]
  tasks: BenchmarkTask[]
  actions: BenchmarkAction[]
  metrics: BenchmarkMetric[]
}
export type EvaluationEvent = {
  event_id?: number
  type: string
  job_id: string
  timestamp: string
  status?: string
  progress?: number
  current_stage?: string | null
  message?: string
  payload?: JsonMap
  terminal?: boolean
}
export type EvaluationJob = {
  job_id: string
  benchmark_id: string
  agent_id: string
  status: string
  created_at?: string
  started_at?: string | null
  finished_at?: string | null
  progress?: number
  current_stage?: string | null
  logs?: string[]
  result?: JsonMap | null
  error?: string | null
}
export type RunRequest = {
  benchmark_id: string
  agent_id: string
  model?: string
  provider?: string
  test_mode?: boolean
  seed?: number
  max_cells?: number
  max_steps?: number
  config_override?: JsonMap
}
export type RunResponse = RunRequest & { job_id: string; status: string }
export type ApiClient = {
  health: () => Promise<Health>
  benchmarks: () => Promise<string[]>
  benchmark: (id: string) => Promise<BenchmarkDetail>
  agents: () => Promise<Agent[]>
  evaluations: () => Promise<EvaluationJob[]>
  evaluation?: (id: string) => Promise<EvaluationJob>
  eventStream?: (id: string) => string
  run: (request: RunRequest) => Promise<RunResponse>
}
type Fetcher = typeof fetch

async function request<T>(fetcher: Fetcher, path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetcher(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) } })
  } catch {
    throw new Error('The evaluation API is unreachable. Start the backend and try again.')
  }
  if (!response.ok) {
    let detail = 'Unknown error'
    try {
      const payload = await response.json() as { detail?: string | JsonMap }
      detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail ?? detail)
    } catch { /* empty response */ }
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
    eventStream: (id) => `/v1/evaluations/${encodeURIComponent(id)}/events`,
    run: async (payload) => {
      try {
        const { test_mode: requestedTestMode, ...requestBody } = payload
        return await request<RunResponse>(fetcher, '/v1/evaluations/run', {
          method: 'POST',
          body: JSON.stringify({
            ...requestBody,
            ...(requestedTestMode ? { test_mode: true } : {}),
            config_override: { ...(payload.config_override ?? {}) },
          }),
        })
      } catch (cause) {
        // A running server may predate the GLM toggle. Do not retry without
        // test_mode: that would silently run a different agent and hide the key issue.
        if (payload.test_mode && cause instanceof Error && cause.message.includes('(422)') && cause.message.includes('test_mode')) {
          throw new Error('The backend is out of date for GLM test mode. Restart the API server, then try again. Your key stays on the backend.')
        }
        throw cause
      }
    },
  }
}
