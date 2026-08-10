import { useEffect, useMemo, useState } from 'react'
import { Activity, Beaker, CheckCircle2, ChevronDown, CircleDot, FlaskConical, Play, RotateCw, ShieldCheck, Sparkles, TerminalSquare } from 'lucide-react'
import { Agent, ApiClient, createApiClient, EvaluationJob, RunResponse } from './api'
import './styles.css'

type Props = { api?: ApiClient }
const defaultApi = createApiClient()
const fallbackBenchmarks = ['pbmc-cell-annotation', 'pbmc-batch-correction', 'pbmc-differential-expression']
const fallbackAgents: Agent[] = ['mock', 'openai', 'anthropic'].map(id => ({ id, type: 'adapter', capabilities: [], available: true }))

function App({ api = defaultApi }: Props) {
  const [benchmarks, setBenchmarks] = useState<string[]>([])
  const [agents, setAgents] = useState<any[]>([])
  const [selectedBenchmark, setSelectedBenchmark] = useState('')
  const [selectedAgent, setSelectedAgent] = useState('')
  const [error, setError] = useState('')
  const [job, setJob] = useState<RunResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [model, setModel] = useState(''); const [provider, setProvider] = useState(''); const [seed, setSeed] = useState(0); const [maxCells, setMaxCells] = useState(''); const [maxSteps, setMaxSteps] = useState('')
  const [recent, setRecent] = useState<EvaluationJob[]>([])
  const [jobDetail, setJobDetail] = useState<EvaluationJob | null>(null)
  const [lastSync, setLastSync] = useState('connecting')

  const load = async () => {
    setError('')
    try {
      const [nextBenchmarks, nextAgents, health, jobs] = await Promise.all([api.benchmarks(), api.agents(), api.health(), api.evaluations()])
      const agentIds = nextAgents.map(agent => typeof agent === 'string' ? agent : agent.id)
      setBenchmarks(nextBenchmarks); setAgents(agentIds); setRecent(jobs.slice(-5).reverse()); setSelectedBenchmark(nextBenchmarks[0] ?? ''); setSelectedAgent(agentIds[0] ?? ''); setLastSync(`online · v${health.version}`)
    } catch (cause) {
    setBenchmarks(fallbackBenchmarks); setAgents(fallbackAgents.map(agent => agent.id)); setSelectedBenchmark(fallbackBenchmarks[0]); setSelectedAgent(fallbackAgents[0].id); setLastSync('offline · demo registry'); setError(cause instanceof Error ? cause.message : 'Unable to connect to API')
    }
  }
  useEffect(() => { void load() }, [])
  useEffect(() => {
    if (!job?.job_id || !api.evaluation) return
    let cancelled = false
    const poll = async () => {
      try {
        const detail = await api.evaluation(job.job_id)
        if (cancelled) return
        setJobDetail(detail)
        if (detail.status === 'PENDING' || detail.status === 'RUNNING' || detail.status === 'pending' || detail.status === 'running') {
          window.setTimeout(() => void poll(), 1500)
        } else {
          setRecent(previous => [detail, ...previous.filter(item => item.job_id !== detail.job_id)].slice(0, 5))
        }
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : 'Unable to read run status')
          window.setTimeout(() => void poll(), 3000)
        }
      }
    }
    void poll()
    return () => { cancelled = true }
  }, [api, job?.job_id])
  const canRun = useMemo(() => Boolean(selectedBenchmark && selectedAgent && !running), [selectedAgent, selectedBenchmark, running])
  const queue = async () => {
    setRunning(true); setError(''); setJob(null)
    try {
      const queued = await api.run({ benchmark_id: selectedBenchmark, agent_id: selectedAgent, model: model || undefined, provider: provider || undefined, seed, max_cells: maxCells ? Number(maxCells) : undefined, max_steps: maxSteps ? Number(maxSteps) : undefined, config_override: {} })
      setJob(queued)
      setJobDetail({ ...queued, result: null })
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to queue evaluation') }
    finally { setRunning(false) }
  }

  return <div className="app-shell">
    <aside className="sidebar" aria-label="Primary navigation">
      <div className="brand"><div className="brand-mark"><FlaskConical size={18} /></div><div><strong>SCAIB</strong><span>scientific evaluation</span></div></div>
      <div className="nav-label">Workspace</div><nav><a className="active" href="#console"><TerminalSquare size={16} /> Console</a><a href="#benchmarks"><Beaker size={16} /> Benchmarks <span className="nav-count">03</span></a><a href="#agents"><Sparkles size={16} /> Agent registry</a><a href="#runs"><Activity size={16} /> Run history</a></nav>
      <div className="sidebar-footer"><div className="secure"><ShieldCheck size={15} /><span>Local evaluation mode</span></div><small>v0.1.0 · API /v1</small></div>
    </aside>
    <main className="main" id="console">
      <header className="topbar"><div className="crumb"><span>WORKSPACE</span><i>/</i><strong>CONSOLE</strong></div><div className="sync"><CircleDot size={11} className={lastSync.startsWith('online') ? 'online' : ''} /> {lastSync}<button className="icon-button" aria-label="Refresh registry" onClick={() => void load()}><RotateCw size={15} /></button></div></header>
      <section className="hero"><div><div className="eyebrow"><span className="pulse" /> SCAIB / EVALUATION CONTROL PLANE</div><h1>Evaluation <em>console</em></h1><p>Configure a reproducible scientific run, inspect the registry, and queue an agent evaluation.</p></div><div className="hero-stamp"><span>SESSION</span><strong>LOCAL-01</strong><small>telemetry enabled</small></div></section>
      {error && <div className="alert" role="alert"><Activity size={16} /><span>{error}</span></div>}
      <section className="metrics" aria-label="System status"><div><span>AVAILABLE BENCHMARKS</span><strong>{benchmarks.length.toString().padStart(2, '0')}</strong><small><CheckCircle2 size={12} /> registry loaded</small></div><div><span>AGENT ADAPTERS</span><strong>{agents.length.toString().padStart(2, '0')}</strong><small><CheckCircle2 size={12} /> provider neutral</small></div><div><span>EXECUTION MODE</span><strong>ASYNC</strong><small><CircleDot size={12} className="online" /> queue enabled</small></div></section>
      <section className="workspace-grid"><div className="panel configure"><div className="panel-head"><div><span className="panel-kicker">01 / CONFIGURE</span><h2>New evaluation run</h2></div><span className="tag">READY</span></div><p className="panel-copy">Select a benchmark and agent adapter. Runs are persisted with their full configuration for replay.</p><label>Benchmark<select aria-label="Benchmark" value={selectedBenchmark} onChange={event => setSelectedBenchmark(event.target.value)}>{benchmarks.map(item => <option key={item}>{item}</option>)}</select></label><label>Agent adapter<select aria-label="Agent adapter" value={selectedAgent} onChange={event => setSelectedAgent(event.target.value)}>{agents.map(item => <option key={item}>{item}</option>)}</select></label><div className="form-grid"><label>Model<input aria-label="Model" value={model} onChange={event => setModel(event.target.value)} placeholder="Optional model" /></label><label>Provider<input aria-label="Provider" value={provider} onChange={event => setProvider(event.target.value)} placeholder="Optional provider" /></label><label>Max cells<input aria-label="Max cells" type="number" min="1" value={maxCells} onChange={event => setMaxCells(event.target.value)} placeholder="All" /></label><label>Max steps<input aria-label="Max steps" type="number" min="1" value={maxSteps} onChange={event => setMaxSteps(event.target.value)} placeholder="Default" /></label></div><div className="config-row"><div><span>DATASET</span><strong>PBMC 68k</strong></div><div><span>REPRODUCIBILITY</span><strong><ShieldCheck size={14} /> Seed {seed}</strong></div></div><button className="primary" disabled={!canRun} onClick={() => void queue()}>{running ? <RotateCw className="spin" size={16} /> : <Play size={16} fill="currentColor" />} {running ? 'Queueing…' : 'Queue evaluation'}<span>↵</span></button></div>
        <div className="right-stack"><div className="panel registry" id="benchmarks"><div className="panel-head"><div><span className="panel-kicker">02 / REGISTRY</span><h2>Benchmark catalog</h2></div><span className="mono">{benchmarks.length} items</span></div><div className="registry-list">{benchmarks.map((item, index) => <div className="registry-item" key={item}><div className="registry-index">0{index + 1}</div><div><strong>{item}</strong><small>{index === 0 ? 'annotation · PBMC reference' : index === 1 ? 'integration · batch correction' : 'expression · differential analysis'}</small></div><ChevronDown size={15} /></div>)}</div></div><div className="panel queue" id="runs"><div className="panel-head"><div><span className="panel-kicker">03 / QUEUE</span><h2>Latest dispatch</h2></div></div>{jobDetail ? <div className="job"><CheckCircle2 size={19} /><div><strong>{jobDetail.status === 'COMPLETED' ? 'Run completed' : jobDetail.status === 'FAILED' ? 'Run failed' : 'Run accepted'}</strong><span>{jobDetail.job_id} · {jobDetail.status}</span></div><span className="job-badge">{jobDetail.status}</span></div> : <div className="empty"><Activity size={18} /><span>{recent.length ? `${recent.length} recent evaluation(s)` : 'No evaluations dispatched in this session.'}</span></div>}{jobDetail?.error && <div className="job-error">{jobDetail.error}</div>}{jobDetail?.result && <div className="result-summary"><span>GLOBAL SCORE</span><strong>{String((jobDetail.result.global_reward as { value?: number } | undefined)?.value ?? '—')}</strong><small>Artifacts and metrics persisted in the run directory.</small></div>}</div></div></section>
      <footer><span>© 2026 SCAIB</span><span>Scientific Agent Capability &amp; Intelligence Benchmark</span><span>All systems nominal <CircleDot size={10} className="online" /></span></footer>
    </main>
  </div>
}
export default App
