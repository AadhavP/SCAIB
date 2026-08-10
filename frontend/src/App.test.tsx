import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

const api: any = { health: vi.fn().mockResolvedValue({ status: 'ok', version: '0.1.0' }), benchmarks: vi.fn().mockResolvedValue(['pbmc-cell-annotation']), agents: vi.fn().mockResolvedValue([{ id: 'mock', type: 'adapter', capabilities: [], available: true }]), evaluations: vi.fn().mockResolvedValue([]), run: vi.fn().mockResolvedValue({ job_id: 'job-1', benchmark_id: 'pbmc-cell-annotation', agent_id: 'mock', status: 'pending' }) }

describe('SCAIB console', () => {
  it('renders the scientific workspace and loaded registry data', async () => {
    render(<App api={api} />)
    expect(screen.getByRole('heading', { name: /evaluation console/i })).toBeInTheDocument()
    expect((await screen.findAllByText('pbmc-cell-annotation')).length).toBeGreaterThan(0)
    expect((await screen.findAllByText('mock')).length).toBeGreaterThan(0)
  })

  it('submits a run and reports its queued job id', async () => {
    const user = userEvent.setup()
    render(<App api={api} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect(await screen.findByText(/job-1/)).toBeInTheDocument()
    expect(api.run).toHaveBeenCalled()
  })

  it('shows an accessible error when registry loading fails', async () => {
    const failedApi = { ...api, benchmarks: vi.fn().mockRejectedValue(new Error('offline')) }
    render(<App api={failedApi} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('offline')
  })

  it('reports dispatch failures without leaving the run button locked', async () => {
    const user = userEvent.setup()
    const failedApi = { ...api, run: vi.fn().mockRejectedValue(new Error('queue unavailable')) }
    render(<App api={failedApi} />)
    await screen.findAllByText('pbmc-cell-annotation')
    await user.click(screen.getByRole('button', { name: /queue evaluation/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('queue unavailable')
    expect(screen.getByRole('button', { name: /queue evaluation/i })).toBeEnabled()
  })
})
