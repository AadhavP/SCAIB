import { describe, expect, it, vi } from 'vitest'
import { createApiClient } from './api'

describe('API client', () => {
  it('loads health, benchmarks, and agents from the typed contract', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok', version: '0.1.0' }) })
    const client = createApiClient(fetcher)
    await expect(client.health()).resolves.toEqual({ status: 'ok', version: '0.1.0' })
    expect(fetcher).toHaveBeenCalledWith('/v1/health', expect.objectContaining({ method: 'GET' }))
  })

  it('rejects non-success responses with a useful error', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: 'unavailable' }) })
    await expect(createApiClient(fetcher).agents()).rejects.toThrow('API request failed (503): unavailable')
  })

  it('builds a replayable evaluation event stream URL', () => {
    expect(createApiClient().eventStream?.('job/1')).toBe('/v1/evaluations/job%2F1/events')
  })

  it('explains when an older backend does not understand GLM test mode', async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: [{ type: 'extra_forbidden', loc: ['body', 'test_mode'], msg: 'Extra inputs are not permitted' }] }),
    })
    await expect(createApiClient(fetcher).run({ benchmark_id: 'pbmc', agent_id: 'mock', test_mode: true }))
      .rejects.toThrow('backend is out of date for GLM test mode')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('posts a run request without mutating the supplied override', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ job_id: 'job-1', benchmark_id: 'pbmc', agent_id: 'mock', status: 'pending' }) })
    const override = { seed: 42 }
    await createApiClient(fetcher).run({ benchmark_id: 'pbmc', agent_id: 'mock', test_mode: false, idempotency_key: 'retry-1', config_override: override })
    expect(fetcher).toHaveBeenCalledWith('/v1/evaluations/run', expect.objectContaining({ method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': 'retry-1' }, body: JSON.stringify({ benchmark_id: 'pbmc', agent_id: 'mock', config_override: { seed: 42 } }) }))
    expect(override).toEqual({ seed: 42 })
  })
})
