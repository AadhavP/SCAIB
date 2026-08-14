import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

const makeApi = () => ({
  health: vi.fn().mockResolvedValue({ status: 'ok', version: '0.1.0' }),
  benchmarks: vi.fn().mockResolvedValue(['pbmc-cell-annotation']),
  benchmark: vi.fn().mockRejectedValue(new Error('metadata unavailable')),
  agents: vi.fn().mockResolvedValue([{ id: 'mock', type: 'adapter', capabilities: ['offline'], available: true }]),
  evaluations: vi.fn().mockResolvedValue([]),
  run: vi.fn().mockResolvedValue({ job_id: 'job-1', benchmark_id: 'pbmc-cell-annotation', agent_id: 'mock', status: 'PENDING' }),
})

describe('SCAIB benchmark flow', () => {
  it('starts with a benchmark catalog and exposes registry data', async () => {
    const api = makeApi()
    render(<App api={api} />)
    expect(screen.getByRole('heading', { name: /evaluation console/i })).toBeInTheDocument()
    expect((await screen.findAllByText('pbmc-cell-annotation')).length).toBeGreaterThan(0)
    await userEvent.setup().click(screen.getByRole('button', { name: /configure benchmark/i }))
    expect(await screen.findByRole('heading', { name: /set up your scientist/i })).toBeInTheDocument()
    expect(await screen.findByText('mock')).toBeInTheDocument()
  })

  it('configures a benchmark, submits a run, and exposes its job id', async () => {
    const user = userEvent.setup()
    const api = makeApi()
    render(<App api={api} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /configure benchmark/i }))
    await user.click(screen.getByRole('checkbox', { name: /glm test mode/i }))
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect((await screen.findAllByText(/job-1/)).length).toBeGreaterThan(0)
    expect(await screen.findByText(/what the model is doing/i)).toBeInTheDocument()
    expect(api.run).toHaveBeenCalledWith(expect.objectContaining({ test_mode: true }))
  })

  it('submits a public endpoint for black-box agent runs', async () => {
    const user = userEvent.setup()
    const api = { ...makeApi(), agents: vi.fn().mockResolvedValue([{ id: 'http-step', type: 'external endpoint', capabilities: ['black-box'], available: true }]) }
    render(<App api={api} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /configure benchmark/i }))
    await user.click(screen.getByRole('button', { name: /http-step/i }))
    await user.type(screen.getByRole('textbox', { name: /agent endpoint/i }), 'https://agent.example/step')
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect(api.run).toHaveBeenCalledWith(expect.objectContaining({ agent_id: 'http-step', agent_endpoint: 'https://agent.example/step' }))
  })

  it('renders the combined headline separately from O, D, and T', async () => {
    const user = userEvent.setup()
    const api = {
      ...makeApi(),
      evaluation: vi.fn().mockResolvedValue({
        job_id: 'job-1', benchmark_id: 'pbmc-cell-annotation', agent_id: 'mock', status: 'COMPLETED', progress: 100,
        request_sha256: 'request-digest', run_id: 'run-1',
        result: { global_reward: { value: 0.95 }, evaluation: { global_agent_score: 0.7, scientific_outcome_score: 0.95, decision_quality_score: 0.8, trajectory_score: 0.9 }, run_id: 'run-1', provenance: { benchmark_specification_digest: 'spec-digest', scoring_profile_sha256: 'profile-digest', archive_manifest_sha256: 'archive-digest', limitations: ['local execution was not filesystem-confined'] } },
      }),
    }
    render(<App api={api} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /configure benchmark/i }))
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect(await screen.findByText('0.700')).toBeInTheDocument()
    expect(screen.getByText('0.950')).toBeInTheDocument()
    expect(screen.getByText('0.800')).toBeInTheDocument()
    expect(screen.getByText('0.900')).toBeInTheDocument()
    await user.click(screen.getByText(/evidence, logs & run details/i))
    expect(screen.getByText(/archive-digest/i)).toBeInTheDocument()
    expect(screen.getByText(/local execution was not filesystem-confined/i)).toBeInTheDocument()
  })

  it('shows an accessible error when registry loading fails', async () => {
    const api = { ...makeApi(), benchmarks: vi.fn().mockRejectedValue(new Error('offline')) }
    render(<App api={api} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('offline')
  })

  it('reports dispatch failures without locking the configuration action', async () => {
    const user = userEvent.setup()
    const api = { ...makeApi(), run: vi.fn().mockRejectedValue(new Error('queue unavailable')) }
    render(<App api={api} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /configure benchmark/i }))
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('queue unavailable')
    expect(screen.getByRole('button', { name: /queue evaluation/i })).toBeEnabled()
  })
})
