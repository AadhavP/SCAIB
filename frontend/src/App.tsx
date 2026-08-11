import { useEffect, useState } from 'react'
import {
  Activity, ArrowLeft, ArrowRight, Beaker, Bot, Check, CheckCircle2, ChevronRight, CircleDot,
  Clock3, Code2, Database, FlaskConical, Gauge, Info, Layers3, MessageSquare, Play, RefreshCw,
  RotateCw, ServerCog, ShieldCheck, Sparkles, TerminalSquare, TriangleAlert, UserRound, XCircle, Zap,
} from 'lucide-react'
import { Agent, ApiClient, BenchmarkAction, BenchmarkDetail, EvaluationEvent, EvaluationJob, JsonMap, RunResponse, createApiClient } from './api'
import './styles.css'

type Props = { api?: ApiClient }
type Screen = 'catalog' | 'configure' | 'run'
type RunState = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'

const defaultApi = createApiClient()
const DEFAULT_AGENT_TURNS = 12
const fallbackBenchmarks = ['pbmc-cell-annotation', 'pbmc-batch-correction', 'pbmc-differential-expression']
const fallbackAgents: Agent[] = [
  { id: 'mock', type: 'deterministic adapter', capabilities: ['reproducible', 'offline'], available: true },
  { id: 'openai', type: 'model runtime', capabilities: ['tool use', 'reasoning'], available: true },
  { id: 'anthropic', type: 'model runtime', capabilities: ['tool use', 'reasoning'], available: true },
]
const fallbackDetails: Record<string, BenchmarkDetail> = {
  'pbmc-cell-annotation': {
    id: 'pbmc-cell-annotation', title: 'PBMC cell-type annotation',
    description: 'Recover biologically meaningful cell types from PBMC expression profiles with a reproducible analysis workflow.',
    version: '1.0.0', tags: ['single-cell', 'annotation', 'PBMC', 'transcriptomics'],
    datasets: [{ id: 'pbmc68k', name: '10x Genomics PBMC 68k', organism: 'Homo sapiens', modality: 'scRNA-seq', expected_observations: { cells: 68579, genes: 32738 } }],
    tasks: [{ id: 'cell-annotation', name: 'PBMC cell-type annotation', description: 'Produce a fully labeled AnnData object with marker-based evidence.', allowed_actions: ['qc', 'normalize', 'marker-genes', 'annotate'], metrics: ['annotation-accuracy', 'annotation-macro-f1', 'runtime'] }],
    actions: [
      { id: 'qc', name: 'Quality control', purpose: 'Filter low-quality cells and calculate QC statistics.' },
      { id: 'normalize', name: 'Normalize expression', purpose: 'Create a comparable expression representation.' },
      { id: 'marker-genes', name: 'Find marker genes', purpose: 'Identify genes that distinguish candidate cell populations.' },
      { id: 'annotate', name: 'Annotate cell types', purpose: 'Assign a supported label to every retained cell.' },
    ],
    metrics: [
      { id: 'annotation-accuracy', name: 'Annotation accuracy', description: 'Cell-level agreement with reference labels.', direction: 'higher_is_better' },
      { id: 'annotation-macro-f1', name: 'Macro F1', description: 'Class-balanced F1 across the cell-type vocabulary.', direction: 'higher_is_better' },
      { id: 'runtime', name: 'Runtime', description: 'Wall-clock execution time.', direction: 'lower_is_better' },
    ],
  },
  'pbmc-batch-correction': {
    id: 'pbmc-batch-correction', title: 'PBMC batch correction',
    description: 'Measure whether an agent can remove technical batch effects while preserving biological structure.',
    version: '1.0.0', tags: ['single-cell', 'integration', 'batch effects'], datasets: [{ id: 'pbmc68k', name: 'PBMC 68k reference', modality: 'scRNA-seq' }],
    tasks: [{ id: 'batch-correction', name: 'Batch correction', description: 'Produce an integrated representation with preserved biology.', allowed_actions: ['qc', 'normalize', 'pca', 'harmony'], metrics: ['batch-removal', 'biology-preservation'] }],
    actions: [{ id: 'qc', name: 'Quality control', purpose: 'Prepare reliable cells for integration.' }, { id: 'normalize', name: 'Normalize expression', purpose: 'Stabilize library-size differences.' }, { id: 'pca', name: 'Build representation', purpose: 'Construct a low-dimensional expression space.' }, { id: 'harmony', name: 'Correct batches', purpose: 'Remove technical variation while preserving biology.' }],
    metrics: [{ id: 'batch-removal', name: 'Batch removal', description: 'How effectively technical variation is reduced.' }, { id: 'biology-preservation', name: 'Biology preservation', description: 'How well biological structure remains intact.' }],
  },
  'pbmc-differential-expression': {
    id: 'pbmc-differential-expression', title: 'PBMC differential expression',
    description: 'Evaluate an agent’s ability to design and execute a defensible differential-expression analysis.',
    version: '1.0.0', tags: ['single-cell', 'expression', 'statistics'], datasets: [{ id: 'pbmc68k', name: 'PBMC 68k reference', modality: 'scRNA-seq' }],
    tasks: [{ id: 'differential-expression', name: 'Differential expression', description: 'Find interpretable genes separating declared cell populations.', allowed_actions: ['qc', 'normalize', 'marker-genes'], metrics: ['marker-quality', 'reproducibility'] }],
    actions: [{ id: 'qc', name: 'Quality control', purpose: 'Remove cells and genes that would bias the comparison.' }, { id: 'normalize', name: 'Normalize expression', purpose: 'Make expression levels comparable.' }, { id: 'marker-genes', name: 'Find marker genes', purpose: 'Rank genes that distinguish the selected groups.' }],
    metrics: [{ id: 'marker-quality', name: 'Marker quality', description: 'Specificity and stability of reported markers.' }, { id: 'reproducibility', name: 'Reproducibility', description: 'Agreement across deterministic reruns.' }],
  },
}

const text = (value: unknown, fallback = '') => typeof value === 'string' && value ? value : fallback
const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null
const titleize = (value: string) => value.replace(/[-_]/g, ' ').replace(/\b\w/g, character => character.toUpperCase())
const statusOf = (value?: string): RunState => {
  const normalized = (value ?? 'PENDING').toUpperCase()
  return ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'].includes(normalized) ? normalized as RunState : 'PENDING'
}
const resultObject = (job: EvaluationJob | null): JsonMap => job?.result && typeof job.result === 'object' ? job.result : {}
const scoreOf = (job: EvaluationJob | null) => {
  const result = resultObject(job)
  const reward = result.global_reward as JsonMap | undefined
  const evaluation = result.evaluation as JsonMap | undefined
  return number(reward?.value) ?? number(evaluation?.global_agent_score) ?? number(evaluation?.benchmark_score)
}
const listOf = (value: unknown): JsonMap[] => Array.isArray(value) ? value.filter(item => Boolean(item) && typeof item === 'object') as JsonMap[] : []

function App({ api = defaultApi }: Props) {
  const [screen, setScreen] = useState<Screen>('catalog')
  const [benchmarks, setBenchmarks] = useState<string[]>(fallbackBenchmarks)
  const [details, setDetails] = useState<Record<string, BenchmarkDetail>>(fallbackDetails)
  const [agents, setAgents] = useState<Agent[]>(fallbackAgents)
  const [selectedBenchmark, setSelectedBenchmark] = useState(fallbackBenchmarks[0])
  const [selectedAgent, setSelectedAgent] = useState('mock')
  const [model, setModel] = useState('')
  const [provider, setProvider] = useState('')
  const [testMode, setTestMode] = useState(false)
  const [starting, setStarting] = useState(false)
  const [seed, setSeed] = useState(42)
  const [maxCells, setMaxCells] = useState('')
  const [maxSteps, setMaxSteps] = useState('')
  const [job, setJob] = useState<RunResponse | null>(null)
  const [jobDetail, setJobDetail] = useState<EvaluationJob | null>(null)
  const [liveEvents, setLiveEvents] = useState<EvaluationEvent[]>([])
  const [streamConnected, setStreamConnected] = useState(false)
  const [recent, setRecent] = useState<EvaluationJob[]>([])
  const [error, setError] = useState('')
  const [lastSync, setLastSync] = useState('connecting')
  const [loadingDetails, setLoadingDetails] = useState(false)

  const benchmark = details[selectedBenchmark] ?? fallbackDetails[selectedBenchmark] ?? fallbackDetails[fallbackBenchmarks[0]]
  const task = benchmark.tasks[0] ?? {}
  const selectedAgentInfo = agents.find(agent => agent.id === selectedAgent)
  const state = statusOf(jobDetail?.status ?? job?.status)
  const isActive = state === 'PENDING' || state === 'RUNNING'

  const load = async () => {
    setError('')
    const [benchmarkResult, agentResult, healthResult, evaluationsResult] = await Promise.allSettled([api.benchmarks(), api.agents(), api.health(), api.evaluations()])
    const failed = [benchmarkResult, agentResult, healthResult, evaluationsResult].find(result => result.status === 'rejected')
    if (benchmarkResult.status === 'fulfilled' && benchmarkResult.value.length) {
      setBenchmarks(benchmarkResult.value)
      setSelectedBenchmark(current => benchmarkResult.value.includes(current) ? current : benchmarkResult.value[0])
    }
    if (agentResult.status === 'fulfilled' && agentResult.value.length) {
      setAgents(agentResult.value)
      setSelectedAgent(current => agentResult.value.some(agent => agent.id === current) ? current : agentResult.value[0].id)
    }
    if (evaluationsResult.status === 'fulfilled') setRecent(evaluationsResult.value.slice(-6).reverse())
    if (healthResult.status === 'fulfilled') setLastSync(`online · v${healthResult.value.version}`)
    else setLastSync('offline · demo registry')
    if (failed) setError(failed.status === 'rejected' && failed.reason instanceof Error ? failed.reason.message : 'Some registry data could not be loaded; showing the built-in catalog.')
  }

  useEffect(() => { void load() }, [])

  useEffect(() => {
    if (!selectedBenchmark || !api.benchmark) return
    let cancelled = false
    setLoadingDetails(true)
    void api.benchmark(selectedBenchmark).then(detail => {
      if (!cancelled) setDetails(previous => ({ ...previous, [selectedBenchmark]: detail }))
    }).catch(() => { /* The built-in summary remains usable when metadata is unavailable. */ }).finally(() => {
      if (!cancelled) setLoadingDetails(false)
    })
    return () => { cancelled = true }
  }, [api, selectedBenchmark])

  useEffect(() => {
    if (!job?.job_id || !api.evaluation) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const detail = await api.evaluation!(job.job_id)
        if (cancelled) return
        setJobDetail(detail)
        const current = statusOf(detail.status)
        if (current === 'PENDING' || current === 'RUNNING') timer = window.setTimeout(() => void poll(), 1200)
        else {
          setRecent(previous => [detail, ...previous.filter(item => item.job_id !== detail.job_id)].slice(0, 6))
        }
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : 'Unable to read run status')
          timer = window.setTimeout(() => void poll(), 3000)
        }
      }
    }
    void poll()
    return () => { cancelled = true; if (timer) window.clearTimeout(timer) }
  }, [api, job?.job_id])

  useEffect(() => {
    if (!job?.job_id || !api.eventStream || typeof EventSource === 'undefined') return
    const source = new EventSource(api.eventStream(job.job_id))
    const handleEvent = (message: MessageEvent<string>) => {
      try {
        const event = JSON.parse(message.data) as EvaluationEvent
        if (event.type === 'heartbeat') return
        setStreamConnected(true)
        setLiveEvents(previous => event.event_id && previous.some(item => item.event_id === event.event_id) ? previous : [...previous, event])
        setJobDetail(previous => previous ? {
          ...previous,
          status: event.status ?? previous.status,
          progress: event.progress ?? previous.progress,
          current_stage: event.current_stage ?? previous.current_stage,
          logs: event.message && !previous.logs?.includes(event.message) ? [...(previous.logs ?? []), event.message] : previous.logs,
        } : previous)
        if (event.terminal) {
          source.close()
          setStreamConnected(false)
        }
      } catch {
        setError('The live event stream returned an invalid event; status polling is still active.')
      }
    }
    source.onopen = () => setStreamConnected(true)
    source.onmessage = handleEvent
    source.onerror = () => setStreamConnected(false)
    return () => {
      source.close()
      setStreamConnected(false)
    }
  }, [api, job?.job_id])

  const chooseBenchmark = (id: string) => {
    setSelectedBenchmark(id)
    setError('')
    setScreen('configure')
  }

  const queue = async () => {
    if (!selectedAgentInfo) {
      setError('Choose an agent before starting the evaluation.')
      return
    }
    if (!selectedAgentInfo.available) {
      setError(`${selectedAgentInfo.id} is unavailable. Choose another agent or enable GLM test mode.`)
      return
    }
    setStarting(true)
    setError('')
    try {
      const queued = await api.run({
        benchmark_id: selectedBenchmark,
        agent_id: selectedAgent,
        model: model || undefined,
        provider: provider || undefined,
        test_mode: testMode,
        seed,
        max_cells: maxCells ? Number(maxCells) : undefined,
        max_steps: maxSteps ? Number(maxSteps) : undefined,
        config_override: {},
      })
      setJob(queued)
      setLiveEvents([])
      setStreamConnected(false)
      setJobDetail({ ...queued, result: null, progress: 0, current_stage: 'Starting evaluation', logs: ['Run accepted by the evaluation queue.'] })
      setScreen('run')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to queue evaluation')
    } finally {
      setStarting(false)
    }
  }

  const resetToCatalog = () => {
    setScreen('catalog')
    setJob(null)
    setJobDetail(null)
    setLiveEvents([])
    setStreamConnected(false)
    setError('')
  }

  return <div className="app-shell">
    <aside className="sidebar" aria-label="Primary navigation">
      <div className="brand"><div className="brand-mark"><FlaskConical size={18} /></div><div><strong>SCAIB</strong><span>scientific evaluation</span></div></div>
      <div className="nav-label">Workspace</div>
      <nav>
        <button className={screen === 'catalog' ? 'active' : ''} onClick={() => setScreen('catalog')}><Beaker size={16} /> Benchmarks <span className="nav-count">{String(benchmarks.length).padStart(2, '0')}</span></button>
        <button className={screen === 'configure' ? 'active' : ''} onClick={() => setScreen('configure')}><Sparkles size={16} /> Configure model</button>
        <button className={screen === 'run' ? 'active' : ''} onClick={() => job && setScreen('run')}><Activity size={16} /> Live run</button>
      </nav>
      <div className="sidebar-footer"><div className="secure"><ShieldCheck size={15} /><span>Local evaluation mode</span></div><small>v0.1.0 · API /v1</small></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="crumb"><span>SCAIB</span><i>/</i><strong>{screen === 'catalog' ? 'BENCHMARKS' : screen === 'configure' ? 'CONFIGURE' : 'LIVE RUN'}</strong></div><div className="sync"><CircleDot size={11} className={lastSync.startsWith('online') ? 'online' : ''} /> {lastSync}<button className="icon-button" aria-label="Refresh registry" onClick={() => void load()}><RefreshCw size={15} /></button></div></header>
      <StepBar screen={screen} />
      {error && <div className="alert" role="alert"><TriangleAlert size={16} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><XCircle size={15} /></button></div>}
      {screen === 'catalog' && <Catalog benchmarks={benchmarks} details={details} selected={selectedBenchmark} onChoose={chooseBenchmark} loading={loadingDetails} recent={recent} />}
      {screen === 'configure' && <Configure benchmark={benchmark} agents={agents} selectedAgent={selectedAgent} setSelectedAgent={setSelectedAgent} model={model} setModel={setModel} provider={provider} setProvider={setProvider} testMode={testMode} setTestMode={setTestMode} seed={seed} setSeed={setSeed} maxCells={maxCells} setMaxCells={setMaxCells} maxSteps={maxSteps} setMaxSteps={setMaxSteps} starting={starting} onBack={resetToCatalog} onRun={() => void queue()} />}
      {screen === 'run' && <RunWalkthrough benchmark={benchmark} task={task} job={jobDetail} events={liveEvents} streamConnected={streamConnected} state={state} isActive={isActive} agent={selectedAgentInfo} model={model} testMode={testMode} seed={seed} maxSteps={maxSteps} onBack={() => setScreen('configure')} onNew={resetToCatalog} />}
      <footer><span>© 2026 SCAIB</span><span>Scientific Agent Capability &amp; Intelligence Benchmark</span><span>API telemetry <CircleDot size={10} className="online" /></span></footer>
    </main>
  </div>
}

function StepBar({ screen }: { screen: Screen }) {
  const steps: [Screen, string][] = [['catalog', 'Choose benchmark'], ['configure', 'Configure model'], ['run', 'Watch evaluation']]
  const current = steps.findIndex(([id]) => id === screen)
  return <div className="stepbar" aria-label="Evaluation setup steps">{steps.map(([id, label], index) => <div className={`step ${index <= current ? 'done' : ''} ${id === screen ? 'current' : ''}`} key={id}><span>{index < current ? <Check size={13} /> : index + 1}</span><strong>{label}</strong>{index < steps.length - 1 && <ChevronRight size={15} />}</div>)}</div>
}

function Catalog({ benchmarks, details, selected, onChoose, loading, recent }: { benchmarks: string[]; details: Record<string, BenchmarkDetail>; selected: string; onChoose: (id: string) => void; loading: boolean; recent: EvaluationJob[] }) {
  return <section className="page catalog-page"><h1 className="sr-only">Evaluation console</h1><div className="page-heading"><div><div className="eyebrow"><span className="pulse" /> STEP 01 / BENCHMARK CATALOG</div><h1>What should we <em>measure?</em></h1><p>Choose a scientific benchmark. Each one is a complete, reproducible workflow with a dataset, observable decisions, and interpretable scores.</p></div><div className="heading-note"><Database size={16} /><span>{benchmarks.length} registered benchmarks<br /><small>Descriptions loaded from the API</small></span></div></div><div className="catalog-grid">{benchmarks.map((id, index) => { const item = details[id] ?? fallbackDetails[id]; return <article className={`benchmark-card ${id === selected ? 'selected' : ''}`} key={id}><div className="card-top"><span className="card-number">0{index + 1}</span><span className="version">v{text(item?.version, '1.0.0')}</span></div><div className="benchmark-icon"><Beaker size={20} /></div><span className="benchmark-id">{id}</span><h2>{text(item?.title, titleize(id))}</h2><p>{text(item?.description, 'A registered scientific evaluation benchmark.')}</p><div className="tag-row">{(item?.tags ?? ['scientific', 'reproducible']).slice(0, 4).map(tag => <span key={tag}>{tag}</span>)}</div><div className="card-meta"><span><Layers3 size={14} /> {item?.tasks.length ?? '—'} task{item?.tasks.length === 1 ? '' : 's'}</span><span><Gauge size={14} /> {item?.metrics.length ?? '—'} metrics</span></div><button className="card-action" aria-label="Configure benchmark" onClick={() => onChoose(id)}>Configure benchmark <ArrowRight size={15} /></button></article>})}</div><div className="catalog-bottom"><div className="info-strip"><Info size={16} /><span><strong>How scoring works</strong> Agents are scored on scientific outcomes, decision quality, trajectory quality, and reproducibility — not just the final answer.</span></div>{recent.length > 0 && <div className="recent-strip"><Clock3 size={15} /> {recent.length} recent run{recent.length === 1 ? '' : 's'} available in this session</div>}</div>{loading && <span className="loading-note">Syncing benchmark metadata…</span>}</section>
}

function Configure({ benchmark, agents, selectedAgent, setSelectedAgent, model, setModel, provider, setProvider, testMode, setTestMode, seed, setSeed, maxCells, setMaxCells, maxSteps, setMaxSteps, starting, onBack, onRun }: { benchmark: BenchmarkDetail; agents: Agent[]; selectedAgent: string; setSelectedAgent: (value: string) => void; model: string; setModel: (value: string) => void; provider: string; setProvider: (value: string) => void; testMode: boolean; setTestMode: (value: boolean) => void; seed: number; setSeed: (value: number) => void; maxCells: string; setMaxCells: (value: string) => void; maxSteps: string; setMaxSteps: (value: string) => void; starting: boolean; onBack: () => void; onRun: () => void }) {
  const task = benchmark.tasks[0] ?? {}
  return <section className="page configure-page"><div className="page-heading compact"><div><div className="eyebrow"><span className="pulse" /> STEP 02 / RUN CONFIGURATION</div><h1>Set up your <em>scientist.</em></h1><p>Everything below is saved with the run, so a score always has a traceable configuration.</p></div></div><div className="configuration-layout"><div className="config-main"><div className="selected-benchmark"><div className="benchmark-icon"><Beaker size={20} /></div><div><span className="micro-label">SELECTED BENCHMARK</span><h2>{benchmark.title}</h2><p>{benchmark.description}</p></div><span className="version">v{benchmark.version}</span></div><div className="form-panel"><div className="section-heading"><div><span className="micro-label">01 / MODEL RUNTIME</span><h2>Who is doing the work?</h2></div><ServerCog size={20} /></div><div className="agent-grid">{agents.map(agent => <button className={`agent-option ${agent.id === selectedAgent ? 'selected' : ''}`} disabled={!agent.available} onClick={() => setSelectedAgent(agent.id)} key={agent.id}><span className="agent-radio">{agent.id === selectedAgent && <Check size={12} />}</span><span><strong>{agent.id}</strong><small>{agent.type}{agent.available ? '' : ' · unavailable'}</small></span><span className="agent-capabilities">{agent.capabilities.slice(0, 2).join(' · ') || 'provider neutral'}</span></button>)}</div><div className="form-grid"><label>Model name<input aria-label="Model" value={model} onChange={event => setModel(event.target.value)} placeholder="e.g. gpt-4.1 or local-model" /><small>Leave blank to use the adapter default.</small></label><label>Provider<input aria-label="Provider" value={provider} onChange={event => setProvider(event.target.value)} placeholder="e.g. openai, anthropic" /><small>Optional runtime/provider hint.</small></label></div><div className="test-mode-card"><label className="toggle-label"><input aria-label="GLM test mode" type="checkbox" checked={testMode} onChange={event => setTestMode(event.target.checked)} /><span className="toggle-switch"><span /></span><span><strong><Zap size={14} /> Use GLM test mode</strong><small>Uses the backend .env credentials and calls the configured GLM/OpenAI-compatible endpoint.</small></span></label></div></div><div className="form-panel"><div className="section-heading"><div><span className="micro-label">02 / REPRODUCIBILITY</span><h2>Control the experiment</h2></div><ShieldCheck size={20} /></div><div className="form-grid three"><label>Random seed<input aria-label="Seed" type="number" min="0" value={seed} onChange={event => setSeed(Number(event.target.value) || 0)} /></label><label>Max cells<input aria-label="Max cells" type="number" min="1" value={maxCells} onChange={event => setMaxCells(event.target.value)} placeholder="All" /><small>Useful for a fast smoke test.</small></label><label>Max steps<input aria-label="Max steps" type="number" min="1" value={maxSteps} onChange={event => setMaxSteps(event.target.value)} placeholder="Default: 12 turns" /></label></div><div className="config-summary"><span><Database size={14} /> {text((benchmark.datasets[0] ?? {}).name, 'Benchmark dataset')}</span><span><TerminalSquare size={14} /> {Array.isArray(task.allowed_actions) ? task.allowed_actions.length : benchmark.actions.length} observable workflow stages</span><span><ShieldCheck size={14} /> Deterministic seed {seed}</span></div></div><div className="action-row"><button className="secondary" onClick={onBack}><ArrowLeft size={15} /> Choose another</button><button className="primary" aria-label="Queue evaluation" disabled={starting} onClick={onRun}>{starting ? <RotateCw size={16} className="spin" /> : <Play size={16} fill="currentColor" />} {starting ? 'Starting…' : 'Start evaluation'} <ArrowRight size={15} /></button></div></div><aside className="workflow-preview"><div className="micro-label">WHAT WILL HAPPEN</div><h2>Evaluation walkthrough</h2><p>The run view will follow each declared stage and reveal the evidence behind every score.</p><div className="preview-list">{benchmark.actions.map((action, index) => <div className="preview-step" key={String(action.id ?? index)}><span>{index + 1}</span><div><strong>{text(action.name, titleize(text(action.id, `Stage ${index + 1}`)))}</strong><small>{text(action.purpose, 'Observable benchmark action')}</small></div></div>)}</div><div className="preview-note"><Sparkles size={15} /><span>Scores appear as soon as the API returns the completed trajectory.</span></div></aside></div></section>
}

function RunWalkthrough({ benchmark, task, job, events, streamConnected, state, isActive, agent, model, testMode, seed, maxSteps, onBack, onNew }: { benchmark: BenchmarkDetail; task: JsonMap; job: EvaluationJob | null; events: EvaluationEvent[]; streamConnected: boolean; state: RunState; isActive: boolean; agent?: Agent; model: string; testMode: boolean; seed: number; maxSteps: string; onBack: () => void; onNew: () => void }) {
  const result = resultObject(job)
  const completedEvents: JsonMap[] = events.filter(event => event.type === 'action_finished' && event.payload?.action_id).map(event => ({
    decision: { action_id: event.payload?.action_id },
    result: event.payload ?? {},
  }))
  const trajectory = [...listOf(result.trajectory), ...completedEvents]
  const activeActionId = [...events].reverse().find(event => event.type === 'action_started' && event.payload?.action_id && !events.some(other => other.type === 'action_finished' && other.payload?.action_id === event.payload?.action_id))?.payload?.action_id
  const logs = job?.logs ?? []
  const progressValue = number(job?.progress) ?? (state === 'COMPLETED' ? 100 : state === 'RUNNING' ? 45 : 8)
  const metrics = listOf(result.final_metrics)
  const evaluation = result.evaluation as JsonMap | undefined
  const domainScores = listOf(evaluation?.domain_scores)
  const score = scoreOf(job)
  const actionTrace = (action: BenchmarkAction) => trajectory.find(step => {
    const decision = step.decision as JsonMap | undefined
    const actionId = text(decision?.action_id, text(step.action_id))
    return actionId === text(action.id)
  })
  const workflow = benchmark.actions.map((action, index) => ({ action, trace: actionTrace(action), index }))
  const configuredStepLimit = maxSteps ? Number(maxSteps) : testMode ? DEFAULT_AGENT_TURNS : null
  // Trust the seed the backend recorded for this job; the local form value can
  // drift from what actually ran once a job is queued.
  const reportedSeed = number((job as JsonMap | null)?.seed) ?? seed
  return <section className="page run-page"><div className="run-header">
<div><div className="eyebrow"><span className={`status-dot ${isActive ? 'active' : state === 'COMPLETED' ? 'complete' : 'failed'}`} /> STEP 03 / LIVE EVALUATION</div><h1>{state === 'COMPLETED' ? <>Evaluation <em>complete.</em></> : state === 'FAILED' ? <>Evaluation <em>stopped.</em></> : <>Your model is <em>working.</em></>}</h1><p>{state === 'COMPLETED' ? 'Review the scorecard and the observable evidence produced by this run.' : state === 'FAILED' ? 'The run stopped, but the failure details and server logs are preserved below.' : 'Follow the declared workflow as the agent inspects data, chooses actions, and earns evidence-backed scores.'}</p></div><div className={`run-status ${state.toLowerCase()}`}><span>{state === 'PENDING' ? 'QUEUED' : state}</span><strong>{Math.round(progressValue)}%</strong><small>{text(job?.current_stage, state === 'PENDING' ? 'Waiting for a worker' : 'Processing benchmark')}</small></div></div><div className="progress-track"><span style={{ width: `${Math.min(100, Math.max(4, progressValue))}%` }} /></div><div className="run-context"><span><Beaker size={14} /> {benchmark.title}</span><span><Sparkles size={14} /> {testMode ? 'GLM test mode' : `${agent?.id ?? job?.agent_id ?? 'agent'}${model ? ` · ${model}` : ''}`}</span><span><ShieldCheck size={14} /> Seed {reportedSeed}</span>{configuredStepLimit !== null && <span><Activity size={14} /> Up to {configuredStepLimit} agent turns</span>}<span className="run-id"><Code2 size={14} /> {job?.job_id ?? 'waiting for job id'}</span></div>{state === 'FAILED' && <div className="failure-card"><TriangleAlert size={20} /><div><strong>Run failed</strong><p>{job?.error ?? 'The server did not provide a failure reason.'}</p><small>Open the evidence panel below for the captured logs and request context.</small></div></div>}<div className="run-layout"><div className="timeline-panel"><div className="panel-title"><div><span className="micro-label">WORKFLOW TRACE</span><h2>What the model is doing</h2></div><span className="live-label"><span className="pulse" /> {streamConnected ? 'LIVE EVENTS' : isActive ? 'LIVE POLLING' : 'FINAL TRACE'}</span></div><div className="timeline">{workflow.map(({ action, trace, index }) => { const traceResult = trace?.result as JsonMap | undefined; const traceStatus = text(traceResult?.status, trace ? 'completed' : index === 0 && isActive ? 'running' : 'waiting'); const failed = traceStatus.toLowerCase().includes('fail'); return <div className={`timeline-item ${trace ? 'visited' : ''} ${failed ? 'failed' : ''}`} key={String(action.id ?? index)}><div className="timeline-marker">{trace ? failed ? <XCircle size={16} /> : <Check size={16} /> : text(activeActionId) === text(action.id) ? <RotateCw size={15} className="spin" /> : index === 0 && isActive ? <RotateCw size={15} className="spin" /> : <span>{index + 1}</span>}</div><div className="timeline-content"><div className="timeline-head"><div><span className="stage-number">STAGE {String(index + 1).padStart(2, '0')}</span><h3>{text(action.name, titleize(text(action.id, 'Workflow stage')))}</h3></div><span className={`stage-status ${trace ? failed ? 'bad' : 'good' : text(activeActionId) === text(action.id) ? 'working' : index === 0 && isActive ? 'working' : ''}`}>{trace ? traceStatus : text(activeActionId) === text(action.id) ? 'in progress' : index === 0 && isActive ? 'in progress' : 'waiting'}</span></div><p>{text(action.purpose, 'The agent will execute this declared benchmark action.')}</p>{trace && <div className="trace-detail"><span><CheckCircle2 size={13} /> observable decision recorded</span>{traceResult?.error ? <span className="trace-error">{text(traceResult.error)}</span> : null}{trace.reward ? <span>local reward {String((trace.reward as JsonMap).value ?? '—')}</span> : null}</div>}</div></div>})}</div>{trajectory.length === 0 && isActive && <div className="waiting-callout"><RotateCw size={17} className="spin" /><span>Waiting for the first observable action from the worker…</span></div>}</div><aside className="score-panel"><div className="panel-title"><div><span className="micro-label">SCORECARD</span><h2>How it is scoring</h2></div><Gauge size={18} /></div><div className="global-score"><span>GLOBAL BENCHMARK SCORE</span><strong>{score === null ? '—' : score.toFixed(3)}</strong><small>{score === null ? 'Available when evaluation completes' : 'combined outcome + decision quality'}</small></div>{domainScores.length > 0 && <div className="score-section"><span className="micro-label">EVALUATION DIMENSIONS</span>{domainScores.map((item, index) => <ScoreBar key={String(item.domain ?? item.name ?? index)} label={titleize(text(item.domain, `Dimension ${index + 1}`))} value={number(item.value)} />)}</div>}{metrics.length > 0 && <div className="score-section"><span className="micro-label">SCIENTIFIC METRICS</span>{metrics.slice(0, 8).map((item, index) => <ScoreBar key={String(item.metric_id ?? item.metric_name ?? index)} label={text(item.metric_name, text(item.metric_id, `Metric ${index + 1}`))} value={number(item.normalized_score) ?? number(item.normalized_value)} />)}</div>}{state === 'COMPLETED' && metrics.length === 0 && <div className="empty-score"><Info size={16} /> The run completed without normalized metric values.</div>}</aside></div><AgentConversation events={events} /><EvidencePanel job={job} logs={logs} task={task} result={result} /><div className="run-actions">{isActive ? <button className="secondary" onClick={onBack}><ArrowLeft size={15} /> Edit configuration</button> : <button className="secondary" onClick={onNew}><Beaker size={15} /> Run another benchmark</button>}{state === 'COMPLETED' && <span className="completion-note"><CheckCircle2 size={16} /> Results persisted with the run artifacts.</span>}</div></section>
}

function AgentConversation({ events }: { events: EvaluationEvent[] }) {
  const messages = events.filter(event => ['agent_planning', 'agent_plan', 'agent_prompt', 'agent_waiting', 'agent_response'].includes(event.type))
  return <section className="conversation-panel">
    <div className="conversation-heading">
      <div className="panel-title"><div><span className="micro-label">AGENT TRANSCRIPT</span><h2><MessageSquare size={17} /> What was asked and answered</h2></div><span className="conversation-count">{messages.length} updates</span></div>
      <p>Public environment observations and structured agent actions. Private chain-of-thought is never captured.</p>
    </div>
    <div className="conversation-list">
      {messages.length === 0 && <div className="conversation-empty"><MessageSquare size={17} /><span>The agent transcript will appear when the first environment observation is sent.</span></div>}
      {messages.map((event, index) => {
        const payload = event.payload ?? {}
        const observation = payload.observation as JsonMap | undefined
        const observationMetadata = observation?.metadata as JsonMap | undefined
        const scenario = observationMetadata?.scenario as JsonMap | undefined
        const availableActions = Array.isArray(observation?.available_actions) ? observation.available_actions.map(String) : []
        const reasoning = payload.reasoning_metadata as JsonMap | undefined
        const plan = payload.plan as JsonMap | undefined
        const planSteps = Array.isArray(plan?.steps) ? plan.steps.map(String) : []
        const successCriteria = Array.isArray(plan?.success_criteria) ? plan.success_criteria.map(String) : []
        const isPlanning = event.type === 'agent_planning'
        const isPlan = event.type === 'agent_plan'
        const isPrompt = event.type === 'agent_prompt'
        const isWaiting = event.type === 'agent_waiting'
        return <article className={`conversation-message ${isPlanning || isPlan ? 'plan' : isPrompt ? 'prompt' : isWaiting ? 'waiting' : 'response'}`} key={String(event.event_id ?? `${event.type}-${index}`)}>
          <div className="conversation-avatar">{isPrompt ? <UserRound size={15} /> : <Bot size={15} />}</div>
          <div className="conversation-content">
            <div className="conversation-meta"><strong>{isPlanning ? 'Runtime' : isPlan ? 'Agent plan' : isPrompt ? 'Environment' : isWaiting ? 'Runtime' : 'Agent'}</strong><span>{isPlan || isPlanning ? 'PLAN' : `STEP ${String(payload.step ?? event.progress ?? index + 1)}`}</span><time>{new Date(event.timestamp).toLocaleTimeString()}</time></div>
            {isPlanning && <p className="typing-line"><span className="typing-dots"><i /><i /><i /></span> Building an overall scientific plan…</p>}
            {isPlan && <><div className="goal-callout"><strong>{text(plan?.goal, 'Scientific benchmark objective')}</strong><p>{text(plan?.adaptation_policy, 'The agent will reassess the plan after each result.')}</p></div>{planSteps.length > 0 && <ol className="plan-steps">{planSteps.map((step, stepIndex) => <li key={`${step}-${stepIndex}`}>{step}</li>)}</ol>}{successCriteria.length > 0 && <div className="plan-criteria"><strong>Success criteria</strong>{successCriteria.map(criteria => <span key={criteria}>{criteria}</span>)}</div>}</>}
            {isPrompt && <><div className="goal-callout"><strong>{text(scenario?.name, 'Scientific task')}</strong><p><b>Goal:</b> {text(scenario?.objective, text(scenario?.description, 'Use the observations to make progress toward the benchmark objective.'))}</p></div>{availableActions.length > 0 && <><span className="conversation-label">LEGAL NEXT ACTIONS</span><div className="action-chips">{availableActions.map(action => <span key={action}>{action}</span>)}</div></>}<details className="conversation-detail"><summary>Inspect observation payload</summary><pre>{JSON.stringify(observation ?? payload, null, 2)}</pre></details></>}            {isWaiting && <p className="typing-line"><span className="typing-dots"><i /><i /><i /></span> Waiting for the model response…</p>}
            {!isPrompt && !isWaiting && !isPlanning && !isPlan && <><p>Selected action: <strong>{text(payload.action_type, 'structured action')}</strong></p>
{Object.keys(payload.parameters as JsonMap ?? {}).length > 0 && <details className="conversation-detail"><summary>Action parameters</summary><pre>{JSON.stringify(payload.parameters, null, 2)}</pre></details>}{reasoning?.summary && <div className="agent-summary">{text(reasoning.summary)}</div>}</>}
          </div>
        </article>
      })}
    </div>
  </section>
}

function ScoreBar({ label, value }: { label: string; value: number | null }) {
  const percent = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100
  return <div className="score-bar"><div><span>{label}</span><strong>{value === null ? '—' : value.toFixed(3)}</strong></div><div className="bar-track"><span style={{ width: `${percent}%` }} /></div></div>
}

function EvidencePanel({ job, logs, task, result }: { job: EvaluationJob | null; logs: string[]; task: JsonMap; result: JsonMap }) {
  const [open, setOpen] = useState(false)
  const trace = listOf(result.trajectory)
  return <details className="evidence-panel" open={open} onToggle={event => setOpen((event.currentTarget as HTMLDetailsElement).open)}><summary><div><TerminalSquare size={16} /><span><strong>Evidence, logs &amp; run details</strong><small>Inspect the exact request, worker messages, and observable trajectory</small></span></div><ChevronRight size={17} /></summary><div className="evidence-grid"><div><span className="micro-label">WORKER LOG</span><div className="log-box">{logs.length ? logs.map((line, index) => <div key={`${line}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span>{line}</div>) : <div className="muted-log">No server log lines were returned. The worker may still be starting.</div>}{job?.error && <div className="log-error">ERROR · {job.error}</div>}</div></div><div><span className="micro-label">RUN CONTEXT</span><pre className="json-box">{JSON.stringify({ job_id: job?.job_id, benchmark_id: job?.benchmark_id, agent_id: job?.agent_id, task_id: task.id, status: job?.status, started_at: job?.started_at, finished_at: job?.finished_at }, null, 2)}</pre></div><div className="trajectory-json"><span className="micro-label">OBSERVABLE TRAJECTORY ({trace.length} events)</span><pre className="json-box">{JSON.stringify(trace, null, 2)}</pre></div></div></details>
}

export default App
