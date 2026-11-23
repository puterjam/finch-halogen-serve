import type * as finch from 'finch';
import { execFile, spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { promisify } from 'node:util';
import { promises as fs } from 'node:fs';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import http from 'node:http';
import os from 'node:os';

// first non-internal IPv4 address = the LAN ip other agents can reach us on
function lanIp(): string | undefined {
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const a of addrs ?? []) {
      if (a.family === 'IPv4' && !a.internal) return a.address;
    }
  }
  return undefined;
}

const execFileAsync = promisify(execFile);

// ---------- settings helpers ----------

interface Config {
  port: number;
  bind: string;
  image: string;
  modelsDir: string;
  containerName: string;
}

// runtime image override (set by update action); loaded from storage at activation
let imageOverride: string | undefined;

// Friendly server options → HALOGEN_* env vars. All flags are read once at
// startup, so changes need a container recreate. The UI edits these options,
// never raw env vars.
interface ServerOptions {
  vision: boolean;                 // HALOGEN_VISION_TOWER=1 (image input)
  qsaForceSel: boolean;            // HALOGEN_QSA_FORCE_SEL=1 (old GPU firmware compat)
  apiKey: string;                  // HALOGEN_API_KEY (empty = no auth)
  thinking: 'default' | 'on' | 'off'; // HALOGEN_ENABLE_THINKING
  reasoningEffort: '' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'; // HALOGEN_REASONING_EFFORT
  kvSlots: number;                 // HALOGEN_KV_SLOTS (1-64, default 4)
  kv8: boolean;                    // HALOGEN_KV8=1 (8-bit KV cache, ~43% less memory)
  kvPool: number;                  // HALOGEN_KV_POOL_POSITIONS (0 = engine default 524288)
  cacheEntries: number;            // HALOGEN_CACHE_ENTRIES (0 = engine default, ~115MB RAM each)
  cacheBranches: number;           // HALOGEN_CACHE_BRANCHES (2 default; 3 for parent+2 subagent fan-out)
  port: number;                    // host API port (container uses the same)
  lan: boolean;                    // bind 0.0.0.0 (LAN reachable) vs 127.0.0.1
  statsApi: boolean;               // serve read-only stats JSON endpoints
  statsApiPort: number;            // stats API listen port
}
const DEFAULT_OPTIONS: ServerOptions = {
  vision: false,
  qsaForceSel: true,
  apiKey: '',
  thinking: 'default',
  reasoningEffort: '',
  kvSlots: 4,
  kv8: false,
  kvPool: 0,
  cacheEntries: 0,
  cacheBranches: 0,
  port: 8731,
  lan: false,
  statsApi: false,
  statsApiPort: 8732,
};
let optionsCache: ServerOptions | undefined;

function getOptions(): ServerOptions {
  return { ...DEFAULT_OPTIONS, ...(optionsCache ?? {}) };
}

function optionsToEnv(o: ServerOptions): { name: string; value: string }[] {
  const env: { name: string; value: string }[] = [];
  if (o.vision) env.push({ name: 'HALOGEN_VISION_TOWER', value: '1' });
  if (o.qsaForceSel) env.push({ name: 'HALOGEN_QSA_FORCE_SEL', value: '1' });
  if (o.apiKey) env.push({ name: 'HALOGEN_API_KEY', value: o.apiKey });
  if (o.thinking === 'on') env.push({ name: 'HALOGEN_ENABLE_THINKING', value: '1' });
  else if (o.thinking === 'off') env.push({ name: 'HALOGEN_ENABLE_THINKING', value: '0' });
  if (o.reasoningEffort) env.push({ name: 'HALOGEN_REASONING_EFFORT', value: o.reasoningEffort });
  const slots = Math.min(Math.max(Math.round(o.kvSlots || 4), 1), 64);
  if (slots !== 4) env.push({ name: 'HALOGEN_KV_SLOTS', value: String(slots) });
  if (o.kv8) env.push({ name: 'HALOGEN_KV8', value: '1' });
  if (o.kvPool) {
    env.push({ name: 'HALOGEN_KV_POOL_POSITIONS', value: String(o.kvPool) });
    // the official 2M-pool recipe on 128 GB machines also caps per-answer tokens
    if (o.kvPool >= 2097152) env.push({ name: 'HALOGEN_MAX_TOK', value: '16384' });
  }
  if (o.cacheEntries) env.push({ name: 'HALOGEN_CACHE_ENTRIES', value: String(o.cacheEntries) });
  if (o.cacheBranches === 3) env.push({ name: 'HALOGEN_CACHE_BRANCHES', value: '3' });
  return env;
}

function envArgs(): string[] {
  return optionsToEnv(getOptions()).flatMap((v) => ['-e', `${v.name}=${v.value}`]);
}

function readConfig(ctx: finch.MiniToolContext): Config {
  const image = imageOverride ?? String(ctx.settings.get('image') ?? 'ghcr.io/peonist-ai/halogen-flash-server:0.18.7');
  const opts = getOptions();
  return {
    port: opts.port || Number(ctx.settings.get('port') ?? 8731),
    bind: opts.lan ? '0.0.0.0' : '127.0.0.1',
    image,
    modelsDir: String(ctx.settings.get('modelsDir') ?? '/home/deck/halogen-models'),
    containerName: String(ctx.settings.get('containerName') ?? 'halogen-flash'),
  };
}

/** Split image reference into repo + tag. e.g. ghcr.io/.../halogen-flash-server:0.18.7 */
function splitImage(image: string): { repo: string; tag: string } {
  const idx = image.lastIndexOf(':');
  if (idx <= 0 || image.indexOf('/', idx) !== -1) return { repo: image, tag: 'latest' };
  return { repo: image.slice(0, idx), tag: image.slice(idx + 1) };
}

// ---------- update detection (GHCR tags/list + GitHub tags fallback) ----------
// GHCR's tags/list index lags behind reality (observed: 0.19.1/0.20.0 pushed
// but missing from tags/list for days), so we also read the GitHub tags API
// (live) and verify candidates against GHCR manifests before advertising.

interface UpdateInfo {
  current: string;
  latest?: string;
  available: boolean;
  checkedAt: number;
  error?: string;
  trace?: string; // diagnostic: which detection path produced the answer
}

let updateCache: UpdateInfo | undefined;
const UPDATE_TTL_MS = 10 * 60 * 1000;

/** semver-ish compare: "0.18.7" vs "0.9.1" → numeric segment compare */
function semverGt(a: string, b: string): boolean {
  const pa = a.split('.').map((n) => parseInt(n, 10) || 0);
  const pb = b.split('.').map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? 0, y = pb[i] ?? 0;
    if (x !== y) return x > y;
  }
  return false;
}

/** GitHub repo tags (live, unlike GHCR tags/list); "v0.20.0" → "0.20.0"
 * Tries the API first, then the tags.atom feed (some runtimes proxy-block api.github.com). */
async function ghRepoTags(ghRepo: string, trace?: string[]): Promise<string[]> {
  // ghRepo may arrive as "ghcr.io/owner/name" — GitHub only wants "owner/name"
  const ghPath = ghRepo.split('/').slice(-2).join('/');
  const clean = (t: string) => t.replace(/^v/, '');
  const valid = (t: string) => /^\d+(\.\d+)+$/.test(t);
  try {
    const res = await fetch(`https://api.github.com/repos/${ghPath}/tags?per_page=30`, {
      headers: { 'User-Agent': 'halogen-serve' },
      signal: AbortSignal.timeout(8000),
    });
    if (res.ok) {
      const data = (await res.json()) as { name: string }[];
      const out = data.map((t) => clean(t.name)).filter(valid);
      trace?.push(`github api ok (${out.length})`);
      return out;
    }
    trace?.push(`github api HTTP ${res.status}`);
  } catch (e) {
    trace?.push(`github api fail: ${(e as Error).name} ${(e as Error).message}`);
  }
  try {
    const res = await fetch(`https://github.com/${ghPath}/tags.atom`, {
      headers: { 'User-Agent': 'halogen-serve' },
      signal: AbortSignal.timeout(8000),
    });
    if (res.ok) {
      const xml = await res.text();
      const out = [...xml.matchAll(/<title>([^<]+)<\/title>/g)].map((m) => clean(m[1].trim())).filter(valid);
      trace?.push(`tags.atom ok (${out.length})`);
      return out;
    }
    trace?.push(`tags.atom HTTP ${res.status}`);
  } catch (e) {
    trace?.push(`tags.atom fail: ${(e as Error).message}`);
  }
  // last resort: curl via shell (the host process's fetch may go through a proxy that 404s github)
  try {
    const { stdout } = await execFileAsync('curl', ['-fsSL', '--max-time', '12', `https://github.com/${ghPath}/tags.atom`], { timeout: 15_000 });
    const out = [...stdout.matchAll(/<title>([^<]+)<\/title>/g)].map((m) => clean(m[1].trim())).filter(valid);
    trace?.push(`curl tags.atom ok (${out.length})`);
    return out;
  } catch (e) {
    trace?.push(`curl tags.atom fail: ${(e as Error).message}`);
  }
  return [];
}

/** does this tag actually exist on GHCR? (manifest HEAD-style probe) */
async function manifestExists(tok: string, repo: string, tag: string, trace?: string[]): Promise<boolean> {
  try {
    const res = await fetch(`https://ghcr.io/v2/${repo}/manifests/${tag}`, {
      headers: {
        Authorization: `Bearer ${tok}`,
        Accept: 'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json',
      },
      signal: AbortSignal.timeout(8000),
    });
    trace?.push(`manifest ${tag}: ${res.status}`);
    return res.ok;
  } catch (e) {
    trace?.push(`manifest ${tag} fail: ${(e as Error).message}`);
    return false;
  }
}

async function checkUpdate(ctx: finch.MiniToolContext, force = false): Promise<UpdateInfo> {
  const cfg = readConfig(ctx);
  const { repo, tag } = splitImage(cfg.image);
  if (!force && updateCache && Date.now() - updateCache.checkedAt < UPDATE_TTL_MS && updateCache.current === tag) {
    return updateCache;
  }
  const info: UpdateInfo = { current: tag, available: false, checkedAt: Date.now() };
  try {
    // GHCR anonymous pull token for the repo
    const tokRes = await fetch(`https://ghcr.io/token?scope=repository:${repo}:pull`, { signal: AbortSignal.timeout(8000) });
    const tok = (await tokRes.json()) as { token?: string };
    if (!tok.token) throw new Error('GHCR token 获取失败');
    const tagRes = await fetch(`https://ghcr.io/v2/${repo}/tags/list`, {
      headers: { Authorization: `Bearer ${tok.token}` },
      signal: AbortSignal.timeout(8000),
    });
    if (!tagRes.ok) throw new Error(`tags/list HTTP ${tagRes.status}`);
    const data = (await tagRes.json()) as { tags?: string[] };
    const versions = (data.tags ?? []).filter((t) => /^\d+(\.\d+)+$/.test(t));
    const trace: string[] = [];
    let latest = tag;
    for (const v of versions) if (semverGt(v, latest)) latest = v;
    trace.push(`ghcr tags/list max=${latest}`);
    // fallback: GHCR tags/list lags; GitHub repo tags are live. Take candidates
    // newer than `latest` and keep the highest one whose manifest really exists.
    if (!semverGt(latest, tag)) {
      const ghTags = (await ghRepoTags(repo, trace)).filter((v) => semverGt(v, latest));
      ghTags.sort((a, b) => (semverGt(a, b) ? -1 : 1));
      trace.push(`candidates=${ghTags.slice(0, 3).join('|') || '-'}`);
      for (const cand of ghTags.slice(0, 3)) {
        if (await manifestExists(tok.token, repo, cand, trace)) { latest = cand; break; }
      }
    }
    info.latest = latest;
    info.available = semverGt(latest, tag);
    info.trace = trace.join('; ');
  } catch (err) {
    info.error = (err as Error).message;
    ctx.logger.warn('update check failed', err);
  }
  updateCache = info;
  return info;
}

// module-level update job so the pull survives panel close/reopen
interface UpdateJob {
  text: string;
  prog?: PullProgress;
  status: 'running' | 'done' | 'error';
  kill?: () => void;
  cancelled?: boolean;
}
let updateJob: UpdateJob | undefined;
const updateListeners = new Set<(j: UpdateJob) => void>();
function emitUpdate() {
  if (updateJob) for (const fn of [...updateListeners]) fn(updateJob);
}

async function updateImage(ctx: finch.MiniToolContext, force = false, onProgress?: (line: string, prog?: PullProgress) => void): Promise<string> {
  if (updateJob?.status === 'running') return '更新已在进行中。';
  const progressCb = (line: string, prog?: PullProgress) => {
    if (updateJob) { updateJob.text = line; updateJob.prog = prog; emitUpdate(); }
    onProgress?.(line, prog);
  };
  const cfg = readConfig(ctx);
  const info = await checkUpdate(ctx, true);
  if (!info.latest) throw new Error(info.error ? `版本检测失败: ${info.error}` : '无法确定最新版本');
  const { repo } = splitImage(cfg.image);
  const newImage = `${repo}:${info.latest}`;
  if (info.latest === info.current && !force) return `已是最新版本（${info.current}）。`;
  updateJob = { text: `开始拉取 ${info.latest}…`, status: 'running' };
  emitUpdate();
  const fmtMB = (n: number) => (n / 1024 / 1024).toFixed(1);
  try {
    await pullImage(newImage, (prog) => {
      const speed = prog.speed ? ` · ${fmtMB(prog.speed)} MB/s` : '';
      progressCb(`拉取 ${info.latest}：${fmtMB(prog.done)} / ${fmtMB(prog.total)} MiB${speed} · 层 ${prog.blobsDone}/${prog.blobsTotal}`, prog);
    }, (kill) => {
      if (updateJob) updateJob.kill = () => { updateJob!.cancelled = true; kill(); };
    });
  } catch (err) {
    if (updateJob?.cancelled) {
      if (updateJob) { updateJob.status = 'done'; updateJob.text = '更新已取消'; updateJob.kill = undefined; emitUpdate(); }
      return '更新已取消。';
    }
    if (updateJob) { updateJob.status = 'error'; updateJob.text = `更新失败: ${(err as Error).message}`; updateJob.kill = undefined; emitUpdate(); }
    throw err;
  }
  // recreate managed container with new image
  await podman(['stop', cfg.containerName]).catch(() => undefined);
  await podman(['rm', '-f', cfg.containerName]).catch(() => undefined);
  const run = await podman([
    'run', '-d', '--name', cfg.containerName,
    '-p', `${cfg.bind}:${cfg.port}:${cfg.port}`,
    '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'keep-groups',
    '--ipc', 'host', '--ulimit', 'memlock=-1:-1',
    ...envArgs(),
    '-v', `${cfg.modelsDir}:/models:ro`,
    newImage,
  ]);
  if (run.code !== 0) throw new Error(`重建容器失败: ${run.stderr || run.stdout}`);
  imageOverride = newImage;
  void ctx.storage.set('imageOverride', newImage);
  updateCache = undefined; // force re-check next time with new current tag
  if (updateJob) { updateJob.status = 'done'; updateJob.text = `已更新到 ${info.latest} 并重启服务（${info.current} → ${info.latest}）。`; updateJob.kill = undefined; emitUpdate(); }
  return `已更新到 ${info.latest} 并重启服务（${info.current} → ${info.latest}）。`;
}

// ---------- shell helpers ----------

// podman pull with parsed per-blob progress (image pulls can take many minutes).
export interface PullProgress {
  pct: number;          // 0-1 aggregated
  done: number;         // bytes
  total: number;        // bytes (sum of known blob sizes)
  speed: number;        // bytes/s
  blobsDone: number;
  blobsTotal: number;
}

async function pullImage(image: string, onProgress?: (p: PullProgress) => void, onChild?: (kill: () => void) => void): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    // podman only emits per-blob progress bars when attached to a TTY;
    // `script -q -e -c ... /dev/null` allocates a pty so we get the same output as a terminal.
    // stty sets the pty width — at width 0 podman suppresses progress bars entirely.
    const p = spawn('script', ['-q', '-e', '-c', `stty cols 120 rows 40; podman pull ${image}`, '/dev/null'], { detached: true });
    const kill = () => { try { process.kill(-p.pid!, 'SIGKILL'); } catch { try { p.kill('SIGKILL'); } catch { /* gone */ } } };
    onChild?.(kill);
    // hold a sleep-inhibit for the whole pull so a locked/suspended laptop
    // doesn't drop the network and kill the download (podman can't resume a partial layer)
    const inhibitor = spawn('systemd-inhibit', ['--what=network:idle:sleep:handle-lid-switch', '--who=halogen-serve', '--why=podman image pull', '--mode=block', 'sleep', '86400'], { stdio: 'ignore' });
    const releaseInhibit = () => { try { inhibitor.kill('SIGTERM'); } catch { /* gone */ } };
    let tail = '';
    let buf = '';
    const UNIT: Record<string, number> = { B: 1, KIB: 1024, MIB: 1024 ** 2, GIB: 1024 ** 3, TIB: 1024 ** 4 };
    const blobs = new Map<string, { done: number; total: number; complete: boolean }>();
    let lastEmit = 0;
    let lastBytes = 0;
    let lastTime = Date.now();
    let speed = 0;
    const onData = (chunk: Buffer) => {
      // strip ANSI escapes (pty output is full of cursor/color codes), split on \r and \n
      const text = chunk.toString().replace(/\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07/g, '');
      tail = (tail + text).slice(-4000);
      buf += text;
      const segs = buf.split(/[\r\n]/);
      buf = segs.pop() ?? '';
      for (const seg of segs) {
        const m = seg.match(/Copying blob (?:sha256:)?([0-9a-f]{8,})\s+(.*)/);
        if (!m) continue;
        const [, id, rest] = m;
        const prev = blobs.get(id);
        if (/^(skipped|done)/.test(rest)) {
          blobs.set(id, { done: prev?.total ?? 0, total: prev?.total ?? 0, complete: true });
        } else {
          const pm = rest.match(/\[[^\]]*\]\s*([\d.]+)\s*([KMGT]i?B|[Bb])\s*\/\s*([\d.]+)\s*([KMGT]i?B|[Bb])/);
          if (pm) blobs.set(id, { done: Number(pm[1]) * (UNIT[pm[2].toUpperCase()] ?? 1), total: Number(pm[3]) * (UNIT[pm[4].toUpperCase()] ?? 1), complete: false });
        }
      }
      const now = Date.now();
      if (onProgress && now - lastEmit > 800) {
        lastEmit = now;
        let done = 0;
        let total = 0;
        let complete = 0;
        for (const b of blobs.values()) { done += b.done; total += b.total; if (b.complete) complete++; }
        if (done > lastBytes && now > lastTime) speed = ((done - lastBytes) / (now - lastTime)) * 1000;
        lastBytes = done;
        lastTime = now;
        onProgress({ pct: total ? Math.min(done / total, 1) : 0, done, total, speed, blobsDone: complete, blobsTotal: blobs.size });
      }
    };
    p.stderr.on('data', onData);
    p.stdout.on('data', onData);
    const watchdog = setTimeout(() => { kill(); reject(new Error('拉取镜像超时（30 分钟）')); }, 1_800_000);
    p.on('error', (err) => { clearTimeout(watchdog); releaseInhibit(); reject(err); });
    p.on('close', (code) => {
      clearTimeout(watchdog);
      releaseInhibit();
      if (code === 0) resolve();
      else reject(new Error(`拉取镜像失败: ${tail.slice(-300)}`));
    });
  });
}

async function podman(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  try {
    const { stdout, stderr } = await execFileAsync('podman', args, { timeout: 60_000, maxBuffer: 8 * 1024 * 1024 });
    return { code: 0, stdout: stdout.trim(), stderr: stderr.trim() };
  } catch (err) {
    const e = err as { code?: number; stdout?: string; stderr?: string; message?: string };
    return { code: typeof e.code === 'number' ? e.code : 1, stdout: (e.stdout ?? '').trim(), stderr: (e.stderr ?? e.message ?? '').trim() };
  }
}

// ---------- status ----------

interface ContainerRow {
  id: string;
  name: string;
  status: string;
  state: string;
  image: string;
}

async function listHalogenContainers(cfg: Config): Promise<ContainerRow[]> {
  const fmt = '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.State}}\t{{.Image}}';
  const r = await podman(['ps', '-a', '--filter', `ancestor=${cfg.image}`, '--format', fmt]);
  const rows: ContainerRow[] = [];
  const parse = (out: string) => out.split('\n').filter(Boolean).map((line) => {
    const [id, name, status, state, image] = line.split('\t');
    return { id: id ?? '', name: name ?? '', status: status ?? '', state: state ?? '', image: image ?? '' };
  });
  rows.push(...parse(r.stdout));
  // also include the managed container even if its image tag differs from settings
  const r2 = await podman(['ps', '-a', '--filter', `name=${cfg.containerName}`, '--format', fmt]);
  for (const row of parse(r2.stdout)) if (!rows.some((x) => x.id === row.id)) rows.push(row);
  return rows;
}

interface HealthInfo {
  reachable: boolean;
  model?: string;
  contextLength?: number;
  latencyMs?: number;
}

async function probeHealth(port: number): Promise<HealthInfo> {
  const t0 = Date.now();
  try {
    const res = await fetch(`http://127.0.0.1:${port}/v1/models`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return { reachable: false };
    const data = (await res.json()) as { data?: { id: string; context_length?: number }[] };
    const first = data.data?.[0];
    return {
      reachable: true,
      model: first?.id,
      contextLength: first?.context_length,
      latencyMs: Date.now() - t0,
    };
  } catch {
    return { reachable: false };
  }
}

// ---- live metrics from /metrics + /cache ----
export interface LiveMetrics {
  // realtime
  decodeTps?: number;
  prefillTps?: number;
  processing?: number;
  deferred?: number;
  kvRatio?: number;      // 0-1
  kvTokens?: number;
  kvPool?: number;
  engineUp?: boolean;
  busyFor?: number;      // seconds oldest in-flight request has run
  queueWaitLongest?: number;
  stale?: boolean;       // true = engine fetch failed, values are the last good snapshot
  staleForMs?: number;
  // cumulative
  requestsTotal?: number;
  promptTokens?: number;
  predictedTokens?: number;
  promptCachedTokens?: number;
  draftAccepted?: number;
  draftProposed?: number;
  cacheHits?: number;
  cacheMisses?: number;
  cacheEntries?: number;
  cacheMaxEntries?: number;
  cacheEvicted?: number;
  promptTokensSaved?: number;
  // power
  gpuW?: number;         // measured amdgpu package power (PPT)
  gpuBusy?: number;      // gpu_busy_percent
  estW?: number;         // estimated whole-machine draw while serving
  energyWh?: number;     // accumulated estimated energy since service start
}

// amdgpu hwmon path is discovered once (index can change between boots)
let powerSensor: string | undefined;
let powerSensorSearched = false;

function findPowerSensor(): string | undefined {
  if (powerSensorSearched) return powerSensor;
  powerSensorSearched = true;
  try {
      const d = readdirSync('/sys/class/hwmon');
      for (const d0 of d) {
        const base = `/sys/class/hwmon/${d0}`;
        let name = '';
        try { name = readFileSync(`${base}/name`, 'utf8').trim(); } catch { continue; }
        if (name !== 'amdgpu') continue;
        for (const f of ['power1_input', 'power1_average']) {
          if (existsSync(`${base}/${f}`)) { powerSensor = `${base}/${f}`; return powerSensor; }
        }
      }
  } catch { /* no hwmon */ }
  return powerSensor;
}

// Idle draw of this machine with the GPU parked (screen off, no inference):
// SoC + RAM + SSD + WiFi measured floor. Used as the base of the estimate.
const IDLE_W = 18;

// energy accumulator (module state: survives panel close)
let energyWh = 0;
let lastPowerAt = 0;
let powerTimer: ReturnType<typeof setInterval> | undefined;

function ensurePowerTimer(): void {
  if (powerTimer) return;
  // keep integrating every 30 s even after the panel closes
  powerTimer = setInterval(() => { readPower(); }, 30000);
  if (typeof powerTimer === 'object' && 'unref' in powerTimer) powerTimer.unref();
}

function readPower(): { gpuW?: number; gpuBusy?: number; estW?: number; energyWh: number } {
  const now = Date.now();
  let gpuW: number | undefined;
  let gpuBusy: number | undefined;
  const p = findPowerSensor();
  if (p) {
    try {
      const v = Number(readFileSync(p, 'utf8').trim());
      if (Number.isFinite(v) && v > 0) gpuW = v / 1e6; // µW → W
    } catch { /* ignore */ }
  }
  try {
    const b = readFileSync('/sys/class/drm/card0/device/gpu_busy_percent', 'utf8').trim();
    if (b) gpuBusy = Number(b);
  } catch { /* ignore */ }
  // Estimated machine draw: measured GPU package power plus a fixed base for
  // everything else (SoC idle, RAM refresh, SSD, WiFi, fans).
  const estW = gpuW !== undefined ? gpuW + IDLE_W : undefined;
  if (estW !== undefined) {
    if (lastPowerAt > 0) {
      const dtH = Math.min((now - lastPowerAt) / 3.6e6, 0.05); // hours, capped
      energyWh += estW * dtH;
    }
    lastPowerAt = now;
  }
  return { gpuW, gpuBusy, estW, energyWh };
}

let lastLive: LiveMetrics | undefined;
let lastLiveAt = 0;

async function getLiveMetrics(port: number): Promise<LiveMetrics | undefined> {
  const power = readPower();
  try {
    const [mRes, cRes] = await Promise.all([
      fetch(`http://127.0.0.1:${port}/metrics`, { signal: AbortSignal.timeout(1500) }),
      fetch(`http://127.0.0.1:${port}/cache`, { signal: AbortSignal.timeout(1500) }).catch(() => undefined),
    ]);
    if (!mRes.ok) throw new Error(`metrics HTTP ${mRes.status}`);
    const text = await mRes.text();
    const g = (name: string): number | undefined => {
      const m = text.match(new RegExp(`^${name.replace(':', ':')}\\s+([\\d.eE+]+)`, 'm'));
      return m ? Number(m[1]) : undefined;
    };
    const out: LiveMetrics = {
      decodeTps: g('llamacpp:predicted_tokens_seconds'),
      prefillTps: g('llamacpp:prompt_tokens_seconds'),
      processing: g('llamacpp:requests_processing'),
      deferred: g('llamacpp:requests_deferred'),
      kvRatio: g('llamacpp:kv_cache_usage_ratio'),
      kvTokens: g('llamacpp:kv_cache_tokens'),
      kvPool: g('halogen:kv_pool_positions'),
      engineUp: g('halogen:engine_up') === 1,
      busyFor: g('halogen:busy_for_seconds'),
      queueWaitLongest: g('halogen:queue_wait_longest_seconds'),
      requestsTotal: g('halogen:requests_total'),
      promptTokens: g('llamacpp:prompt_tokens_total'),
      predictedTokens: g('llamacpp:tokens_predicted_total'),
      promptCachedTokens: g('halogen:prompt_tokens_cached_total'),
      draftAccepted: g('halogen:draft_tokens_accepted_total'),
      draftProposed: g('halogen:draft_tokens_total'),
    };
    if (cRes?.ok) {
      const c = (await cRes.json()) as Record<string, number>;
      out.cacheHits = c.hits;
      out.cacheMisses = c.misses;
      out.cacheEntries = c.entries;
      out.cacheMaxEntries = c.max_entries;
      out.cacheEvicted = c.evicted;
      out.promptTokensSaved = c.prompt_tokens_saved;
    }
    Object.assign(out, power);
    ensurePowerTimer();
    lastLive = out;
    lastLiveAt = Date.now();
    return out;
  } catch {
    // engine busy (e.g. mid-prefill) or down: serve the last good snapshot so
    // 1 s pollers always get data; power is still fresh from this call
    if (lastLive) {
      ensurePowerTimer();
      return { ...lastLive, ...power, stale: true, staleForMs: Date.now() - lastLiveAt };
    }
    return undefined;
  }
}

interface StatusSnapshot {
  running: boolean;
  managed: boolean; // our named container exists
  container?: ContainerRow;
  foreign?: ContainerRow[]; // other halogen containers not managed by us
  health: HealthInfo;
  port: number;
  bind: string;
  lanIp?: string;
  image: string;
  containerName: string;
  modelsDir: string;
  checkedAt: string;
}

async function getStatus(ctx: finch.MiniToolContext): Promise<StatusSnapshot> {
  const cfg = readConfig(ctx);
  const rows = await listHalogenContainers(cfg);
  const managed = rows.find((r) => r.name === cfg.containerName);
  const foreign = rows.filter((r) => r.name !== cfg.containerName);
  const running = managed ? managed.state === 'running' : foreign.some((r) => r.state === 'running');
  const health = await probeHealth(cfg.port);
  return {
    running: running || health.reachable,
    managed: Boolean(managed),
    container: managed,
    foreign: foreign.length ? foreign : undefined,
    health,
    port: cfg.port,
    bind: cfg.bind,
    lanIp: cfg.bind === '0.0.0.0' ? lanIp() : undefined,
    image: cfg.image,
    containerName: cfg.containerName,
    modelsDir: cfg.modelsDir,
    checkedAt: new Date().toISOString(),
  };
}

// ---------- actions ----------

async function startService(ctx: finch.MiniToolContext): Promise<string> {
  const cfg = readConfig(ctx);
  // prefer the newest halogen image present locally (e.g. after a manual
  // `podman pull` that the tool did not do itself)
  const newest = await newestLocalImage();
  const image = newest ?? cfg.image;
  if (newest && newest !== cfg.image) {
    imageOverride = newest;
    await ctx.storage.set('imageOverride', newest).catch(() => undefined);
  }
  const rows = await listHalogenContainers(cfg);
  const mine = rows.find((r) => r.name === cfg.containerName);
  if (mine && mine.state === 'running') return `容器 ${cfg.containerName} 已在运行。`;
  if (mine && (image === cfg.image || (mine.image ?? '').includes(splitImage(image).tag))) {
    const r = await podman(['start', cfg.containerName]);
    if (r.code !== 0) throw new Error(r.stderr || r.stdout);
    return `已启动已有容器 ${cfg.containerName}。`;
  }
  if (mine) {
    // existing container runs an older image: recreate on the newest one
    await podman(['stop', cfg.containerName]).catch(() => undefined);
    await podman(['rm', '-f', cfg.containerName]).catch(() => undefined);
  }
  const r = await podman([
    'run', '-d', '--name', cfg.containerName,
    '-p', `${cfg.bind}:${cfg.port}:${cfg.port}`,
    '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'keep-groups',
    '--ipc', 'host', '--ulimit', 'memlock=-1:-1',
    ...envArgs(),
    '-v', `${cfg.modelsDir}:/models:ro`,
    image,
  ]);
  if (r.code !== 0) throw new Error(r.stderr || r.stdout);
  return mine
    ? `已用最新镜像 ${splitImage(image).tag} 重建并启动容器 ${cfg.containerName}。`
    : `已创建并启动容器 ${cfg.containerName}（端口 ${cfg.port}）。`;
}

/** highest-version halogen image tag present in the local store */
async function newestLocalImage(): Promise<string | undefined> {
  const r = await podman(['images', '--format', '{{.Repository}}:{{.Tag}}']);
  if (r.code !== 0) return undefined;
  let best: string | undefined;
  let bestTag = '';
  for (const line of r.stdout.split('\n')) {
    const ref = line.trim();
    if (!ref.startsWith('ghcr.io/peonist-ai/halogen-flash-server:')) continue;
    const tag = splitImage(ref).tag;
    if (!/^\d+(\.\d+)+$/.test(tag)) continue;
    if (!best || semverGt(tag, bestTag)) { best = ref; bestTag = tag; }
  }
  return best;
}

async function stopService(ctx: finch.MiniToolContext): Promise<string> {
  const cfg = readConfig(ctx);
  const r = await podman(['stop', cfg.containerName]);
  if (r.code !== 0) {
    // maybe a foreign container is running
    const rows = await listHalogenContainers(cfg);
    const runningForeign = rows.find((x) => x.state === 'running');
    if (runningForeign) {
      const r2 = await podman(['stop', runningForeign.name]);
      if (r2.code !== 0) throw new Error(r2.stderr || r2.stdout);
      return `已停止容器 ${runningForeign.name}（非本工具命名容器）。`;
    }
    throw new Error(r.stderr || r.stdout || '未找到容器');
  }
  return `已停止容器 ${cfg.containerName}。`;
}

async function removeService(ctx: finch.MiniToolContext): Promise<string> {
  const cfg = readConfig(ctx);
  await podman(['stop', cfg.containerName]).catch(() => undefined);
  const r = await podman(['rm', '-f', cfg.containerName]);
  if (r.code !== 0) throw new Error(r.stderr || r.stdout || '未找到容器');
  return `已删除容器 ${cfg.containerName}（下次启动将按当前设置重新创建）。`;
}

async function recreateService(ctx: finch.MiniToolContext): Promise<string> {
  const cfg = readConfig(ctx);
  await podman(['stop', cfg.containerName]).catch(() => undefined);
  await podman(['rm', '-f', cfg.containerName]).catch(() => undefined);
  const r = await podman([
    'run', '-d', '--name', cfg.containerName,
    '-p', `${cfg.bind}:${cfg.port}:${cfg.port}`,
    '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'keep-groups',
    '--ipc', 'host', '--ulimit', 'memlock=-1:-1',
    ...envArgs(),
    '-v', `${cfg.modelsDir}:/models:ro`,
    cfg.image,
  ]);
  if (r.code !== 0) throw new Error(r.stderr || r.stdout);
  return `已按当前设置重建并启动容器 ${cfg.containerName}。`;
}

async function getLogs(ctx: finch.MiniToolContext, lines: number, withTimestamps = false): Promise<string> {
  const cfg = readConfig(ctx);
  let target = cfg.containerName;
  const rows = await listHalogenContainers(cfg);
  if (!rows.some((r) => r.name === target)) {
    const any = rows.find((r) => r.state === 'running') ?? rows[0];
    if (!any) return '（没有 halogen 容器，无法读取日志）';
    target = any.name;
  }
  const r = await podman(['logs', ...(withTimestamps ? ['-t'] : []), '--tail', String(lines), target]);
  const out = [r.stdout, r.stderr].filter(Boolean).join('\n');
  return out || '（日志为空）';
}

// ---------- performance stats (parsed from serve_api summary lines) ----------

interface ReqStat {
  t?: number;                // epoch ms from podman log timestamp
  decodeTps?: number;      // mtp X tok in Ys = Z t/s (generation speed)
  genTokens?: number;
  commitPerRound?: number; // mtp accept rate
  promptTokens?: number;
  cachedPct?: number;      // prompt cache hit %
  prefillTps?: number;     // full-prefill rate (only when no cache shortcut)
  prefillSecs?: number;
  newTokens?: number;      // tokens actually prefilled (after cache)
}

function parseStats(logText: string): ReqStat[] {
  const stats: ReqStat[] = [];
  for (const line of logText.split('\n')) {
    if (!line.includes('serve_api: mtp')) continue;
    const s: ReqStat = {};
    const ts = line.match(/^(\d{4}-\d{2}-\d{2}T[\d:.+]+Z?)\s/);
    if (ts) { const ms = Date.parse(ts[1]); if (isFinite(ms)) s.t = ms; }
    let m = line.match(/mtp (\d+) tok in [\d.]+s = ([\d.]+) t\/s/);
    if (m) { s.genTokens = Number(m[1]); s.decodeTps = Number(m[2]); }
    m = line.match(/commit ([\d.]+)\/round/);
    if (m) s.commitPerRound = Number(m[1]);
    m = line.match(/prompt (\d+)(?: \((\d+) cached, ([\d.]+)%\))?/);
    if (m) {
      s.promptTokens = Number(m[1]);
      if (m[3]) s.cachedPct = Number(m[3]);
    }
    m = line.match(/prefill ([\d.]+)s = ([\d.]+) t\/s/);
    if (m) { s.prefillSecs = Number(m[1]); s.prefillTps = Number(m[2]); }
    else {
      m = line.match(/prefill ([\d.]+)s \((\d+) new\)/);
      if (m) { s.prefillSecs = Number(m[1]); s.newTokens = Number(m[2]); }
    }
    stats.push(s);
  }
  return stats;
}

function avg(nums: (number | undefined)[]): number | undefined {
  const v = nums.filter((n): n is number => typeof n === 'number' && isFinite(n));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined;
}

function fmt(n: number | undefined, digits = 1): string {
  return n === undefined ? '—' : n.toFixed(digits);
}

// effective prefill rate: cold = reported t/s; warm = new tokens / secs
function effPrefillTps(s: ReqStat): number | undefined {
  if (s.prefillTps !== undefined) return s.prefillTps;
  if (s.newTokens !== undefined && s.prefillSecs && s.prefillSecs > 0) return s.newTokens / s.prefillSecs;
  return undefined;
}

interface StatsPayload {
  count: number;
  points: { t?: number; decode?: number; prefill?: number; cached?: number; prompt?: number }[];
  summary: {
    decodeAvg?: number; decodePeak?: number;
    prefillColdAvg?: number; prefillWarmAvg?: number;
    cacheAvg?: number; commitAvg?: number;
    totalGenTokens?: number;
  };
}

// ---------- read-only stats API for remote agents ----------
// Endpoints (JSON): /live (instant metrics + power), /stats?lines=N (log-parsed
// per-request stats), /status (container + health), /all (aggregate of all three).
// Listens on 0.0.0.0 when the LAN option is on, otherwise localhost only.

let statsServer: http.Server | undefined;
let statsServerPort = 0;

async function statsSnapshot(ctx: finch.MiniToolContext, path: string, lines: number): Promise<unknown> {
  const port = getOptions().port;
  if (path === '/live') return { ok: true, ts: Date.now(), live: await getLiveMetrics(port) };
  if (path === '/stats') return { ok: true, ts: Date.now(), stats: await getStatsData(ctx, lines) };
  if (path === '/status') {
    const s = await getStatus(ctx);
    return {
      ok: true, ts: Date.now(),
      running: s.running, image: s.image, port: s.port,
      lanUrl: s.lanIp ? `http://${s.lanIp}:${s.port}/v1` : undefined,
      model: s.health.model, contextLength: s.health.contextLength,
      latencyMs: s.health.latencyMs,
    };
  }
  // /all — one round trip gets everything a remote dashboard needs
  const [live, stats, status] = await Promise.all([
    getLiveMetrics(port),
    getStatsData(ctx, lines),
    getStatus(ctx),
  ]);
  return { ok: true, ts: Date.now(), live, stats, status: { running: status.running, image: status.image, model: status.health.model } };
}

function syncStatsApi(ctx: finch.MiniToolContext): void {
  const o = getOptions();
  const wantPort = o.statsApi ? o.statsApiPort : 0;
  if (statsServer && statsServerPort === wantPort && wantPort !== 0) return;
  if (statsServer) { statsServer.close(); statsServer = undefined; statsServerPort = 0; }
  if (!wantPort) return;
  const bind = o.lan ? '0.0.0.0' : '127.0.0.1';
  const srv = http.createServer(async (req, res) => {
    const send = (code: number, body: unknown) => {
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify(body));
    };
    try {
      const u = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
      const path = u.pathname.replace(/\/+$/, '') || '/';
      if (path === '/' || path === '') {
        return send(200, { ok: true, name: 'halogen-serve stats api', endpoints: ['/live', '/stats?lines=N', '/status', '/all'] });
      }
      if (!['/live', '/stats', '/status', '/all'].includes(path)) return send(404, { ok: false, error: 'not found' });
      const lines = Math.min(Math.max(Number(u.searchParams.get('lines') ?? 2000) || 2000, 100), 5000);
      send(200, await statsSnapshot(ctx, path, lines));
    } catch (err) {
      send(500, { ok: false, error: (err as Error).message });
    }
  });
  srv.on('error', (err) => { ctx.logger.warn('stats api listen failed', err); statsServer = undefined; statsServerPort = 0; });
  srv.listen(wantPort, bind, () => ctx.logger.info(`halogen stats api on http://${bind}:${wantPort}`));
  statsServer = srv;
  statsServerPort = wantPort;
}

function buildStats(stats: ReqStat[]): StatsPayload {
  const recent = stats.slice(-120);
  const cold = recent.filter((s) => s.prefillTps !== undefined);
  const warm = recent.filter((s) => s.prefillTps === undefined && s.newTokens !== undefined && s.prefillSecs !== undefined && s.prefillSecs! > 0);
  const cached = recent.filter((s) => s.cachedPct !== undefined);
  const commit = recent.filter((s) => s.commitPerRound !== undefined);
  return {
    count: recent.length,
    points: recent.map((s) => ({
      t: s.t,
      decode: s.decodeTps,
      prefill: effPrefillTps(s),
      cached: s.cachedPct,
      prompt: s.promptTokens,
    })),
    summary: {
      decodeAvg: avg(recent.map((s) => s.decodeTps)),
      decodePeak: recent.some((s) => s.decodeTps !== undefined) ? Math.max(...recent.flatMap((s) => (s.decodeTps !== undefined ? [s.decodeTps] : []))) : undefined,
      prefillColdAvg: avg(cold.map((s) => s.prefillTps)),
      prefillWarmAvg: avg(warm.map((s) => effPrefillTps(s))),
      cacheAvg: avg(cached.map((s) => s.cachedPct)),
      commitAvg: avg(commit.map((s) => s.commitPerRound)),
      totalGenTokens: recent.reduce((a, s) => a + (s.genTokens ?? 0), 0),
    },
  };
}

async function getStatsData(ctx: finch.MiniToolContext, tail: number): Promise<StatsPayload> {
  const raw = await getLogs(ctx, tail, true);
  return buildStats(parseStats(raw));
}

async function getStats(ctx: finch.MiniToolContext, tail: number): Promise<string> {
  const raw = await getLogs(ctx, tail, true);
  const stats = parseStats(raw);
  if (!stats.length) return '日志里还没有性能统计（服务处理过请求后才会出现 serve_api: mtp … 行）';
  const p = buildStats(stats);
  const s = p.summary;
  const lines = [
    `最近 ${p.count} 个请求：`,
    `  生成速度：平均 ${fmt(s.decodeAvg)} tok/s，峰值 ${fmt(s.decodePeak)} tok/s`,
  ];
  if (s.prefillColdAvg !== undefined) lines.push(`  冷 prefill（无缓存命中）：平均 ${fmt(s.prefillColdAvg)} tok/s`);
  if (s.prefillWarmAvg !== undefined) lines.push(`  增量 prefill（缓存后新 token）：平均 ${fmt(s.prefillWarmAvg)} tok/s`);
  if (s.cacheAvg !== undefined) lines.push(`  提示缓存命中率：平均 ${fmt(s.cacheAvg)}%`);
  if (s.commitAvg !== undefined) lines.push(`  MTP 接受率：平均 ${fmt(s.commitAvg, 2)} tok/轮`);
  const last = stats[stats.length - 1];
  lines.push(`最新一次：生成 ${fmt(last.decodeTps)} tok/s，提示 ${last.promptTokens ?? '—'} tok${last.cachedPct !== undefined ? `（缓存 ${last.cachedPct}%）` : ''}${effPrefillTps(last) !== undefined ? `，prefill ${fmt(effPrefillTps(last))} tok/s` : ''}`);
  return lines.join('\n');
}

// ---------- live log streaming ----------

class LogStreamer {
  private proc?: ChildProcessWithoutNullStreams;
  private buf = '';
  private timer?: NodeJS.Timeout;

  constructor(
    private readonly send: (msg: unknown) => Promise<void> | void,
  ) {}

  async start(ctx: finch.MiniToolContext, tail: number): Promise<void> {
    this.stop();
    const cfg = readConfig(ctx);
    let target = cfg.containerName;
    const rows = await listHalogenContainers(cfg);
    if (!rows.some((r) => r.name === target)) {
      const any = rows.find((r) => r.state === 'running') ?? rows[0];
      if (!any) { await this.send({ type: 'log-info', text: '（没有 halogen 容器）' }); return; }
      target = any.name;
    }
    await this.send({ type: 'log-clear' });
    await this.send({ type: 'log-info', text: `── 正在跟踪 ${target} 的实时日志（最近 ${tail} 行起）──` });
    const proc = spawn('podman', ['logs', '-f', '--tail', String(tail), target]);
    this.proc = proc;
    const onData = (d: Buffer) => {
      this.buf += d.toString('utf8');
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
    proc.on('error', (e) => { void this.send({ type: 'log-info', text: `日志进程错误: ${e.message}` }); });
    proc.on('exit', (code) => {
      if (this.proc === proc) {
        this.proc = undefined;
        void this.send({ type: 'log-info', text: `── 日志跟踪结束（exit ${code}）──` });
      }
    });
    // batch lines to the panel every 500 ms
    this.timer = setInterval(() => {
      if (!this.buf) return;
      const chunk = this.buf;
      this.buf = '';
      void this.send({ type: 'log-chunk', text: chunk });
    }, 500);
  }

  stop(): void {
    if (this.timer) { clearInterval(this.timer); this.timer = undefined; }
    if (this.proc) { const p = this.proc; this.proc = undefined; p.kill('SIGKILL'); }
    this.buf = '';
  }
}

// ---------- memory ----------

interface MemInfo { totalGB: number; availableGB: number }

async function readMemInfo(): Promise<MemInfo | undefined> {
  try {
    const txt = await fs.readFile('/proc/meminfo', 'utf8');
    const total = /^MemTotal:\s+(\d+) kB$/m.exec(txt);
    const avail = /^MemAvailable:\s+(\d+) kB$/m.exec(txt);
    if (!total) return undefined;
    return {
      totalGB: Math.round(Number(total[1]) / 1024 / 1024 * 10) / 10,
      availableGB: avail ? Math.round(Number(avail[1]) / 1024 / 1024 * 10) / 10 : 0,
    };
  } catch {
    return undefined;
  }
}

// Official: weights pin ~62 GiB (68 with w4b) + KV pool reserved up front.
// Reference host is 128 GB ("holds most of a 128 GB host"); 96 GB is the floor.
// Note: a "128 GB" host reports ~124.9 GiB in /proc/meminfo (GPU carve-out),
// so the recommended threshold allows for that.
const RAM_MIN_GB = 96;
const RAM_RECOMMENDED_GB = 120; // ≈ a 128 GB host as reported by meminfo

// ---------- environment check ----------

interface CheckItem { ok: boolean; label: string; detail?: string; updateAvailable?: boolean }

async function checkEnv(ctx: finch.MiniToolContext): Promise<CheckItem[]> {
  const cfg = readConfig(ctx);
  const items: CheckItem[] = [];

  // podman
  const pv = await execFileAsync('podman', ['--version'], { timeout: 10_000 }).catch(() => undefined);
  items.push(pv ? { ok: true, label: 'podman', detail: pv.stdout.trim() } : { ok: false, label: 'podman', detail: '未安装或不可用' });

  // image
  const img = await podman(['image', 'exists', cfg.image]);
  items.push({ ok: img.code === 0, label: '容器镜像', detail: img.code === 0 ? cfg.image : `${cfg.image} 不存在，需 podman pull` });

  // models dir
  try {
    const files = await fs.readdir(cfg.modelsDir);
    const hgn = files.filter((f) => f.endsWith('.hgn'));
    items.push({ ok: hgn.length > 0, label: '模型目录', detail: `${cfg.modelsDir}（${hgn.length} 个 .hgn 文件）` });
  } catch {
    items.push({ ok: false, label: '模型目录', detail: `${cfg.modelsDir} 不存在` });
  }

  // GPU devices
  for (const dev of ['/dev/kfd', '/dev/dri']) {
    const ok = await fs.access(dev).then(() => true).catch(() => false);
    items.push({ ok, label: `GPU 设备 ${dev}`, detail: ok ? '存在' : '缺失（ROCm GPU 直通不可用）' });
  }

  // RAM: weights pin ~62 GiB (68 with w4b) + KV pool; official reference host is 128 GB
  const mem = await readMemInfo();
  if (mem) {
    const ok = mem.totalGB >= RAM_MIN_GB;
    const rec = mem.totalGB >= RAM_RECOMMENDED_GB;
    items.push({
      ok: rec ? true : ok,
      label: `内存（要求 ≥${RAM_MIN_GB} GB，推荐 128 GB）`,
      detail: `本机 ${mem.totalGB} GB，当前可用 ${mem.availableGB} GB${rec ? '' : ok ? '（低于推荐值，可运行但余量紧张）' : ' — 低于下限，权重约锁 62 GiB 内存，可能无法启动或运行中卡死'}`,
    });
  } else {
    items.push({ ok: false, label: '内存', detail: '无法读取 /proc/meminfo' });
  }

  // port / service
  const health = await probeHealth(cfg.port);
  items.push(health.reachable
    ? { ok: true, label: `API 端口 ${cfg.port}`, detail: `服务在线${health.model ? `，模型 ${health.model}` : ''}` }
    : { ok: false, label: `API 端口 ${cfg.port}`, detail: '服务未响应（未启动或仍在加载模型）' });

  // image update (cached check, 10-min TTL)
  const upd = await checkUpdate(ctx);
  if (upd.error) {
    items.push({ ok: true, label: '镜像更新', detail: `检测失败：${upd.error}` });
  } else if (upd.available) {
    items.push({ ok: true, label: '镜像更新', detail: `发现新版本 ${upd.latest}（当前 ${upd.current}），可更新`, updateAvailable: true });
  } else {
    items.push({ ok: true, label: '镜像更新', detail: `已是最新版本（${upd.latest || upd.current}）${upd.trace ? ` · ${upd.trace}` : ''}` });
  }

  return items;
}

// ---------- text rendering ----------

function renderStatus(s: StatusSnapshot): string {
  const lines = [
    `服务状态: ${s.running ? '🟢 运行中' : '⚪ 已停止'}`,
    `API: http://127.0.0.1:${s.port}/v1${s.health.reachable && s.health.model ? `（模型 ${s.health.model}，上下文 ${s.health.contextLength ?? '?'}，延迟 ${s.health.latencyMs}ms）` : ''}`,
    `容器: ${s.container ? `${s.containerName} [${s.container.state}] ${s.container.status}` : '未创建（首次启动时自动创建）'}`,
  ];
  if (s.foreign?.length) lines.push(`⚠ 发现其他 halogen 容器: ${s.foreign.map((f) => `${f.name}[${f.state}]`).join(', ')}`);
  return lines.join('\n');
}

// ---------- activation ----------

export async function activate(ctx: finch.MiniToolContext): Promise<void> {
  ctx.logger.info('halogen-serve activated');

  // restore image override from a previous update
  imageOverride = await ctx.storage.get<string>('imageOverride') ?? undefined;
  optionsCache = await ctx.storage.get<ServerOptions>('serverOptions') ?? undefined;
  if (optionsCache) delete (optionsCache as Partial<ServerOptions>).apiKey; // key lives in secrets
  const savedKey = await ctx.secrets.get('halogen.apiKey').catch(() => undefined);
  if (savedKey && optionsCache) optionsCache.apiKey = savedKey;

  // read-only stats API (honours the saved toggle)
  syncStatsApi(ctx);

  // custom icons (lucide SVGs not in Finch built-in list)
  ctx.icons.register('halogen-icons', {
    'refresh': { svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>' },
    'power': { svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v10"/><path d="M18.4 6.6a9 9 0 1 1-12.77.04"/></svg>' },
  });

  // ---- Agent tool ----
  ctx.subscriptions.push(ctx.tools.register({
    name: 'halogen_serve_control',
    title: 'Halogen Serve',
    description: `Manage the local Halogen model inference server (podman container, OpenAI-compatible API on localhost).
action:
  status  — show container state, API health, model info
  start   — start the service (creates the container on first run)
  stop    — stop the service
  restart — stop then start the service
  remove  — delete the managed container so next start recreates it with current settings
  logs    — show recent container logs (param: lines, default 80)
  check   — run environment checks (podman, image, models dir, GPU devices, port)
  stats   — parse request performance from logs: decode tok/s, prefill tok/s, prompt-cache hit rate (param: lines, default 2000)
  update  — check GHCR for a newer image tag; pull it and recreate the container (param: force=true to pull even if tags match)`,
    inputSchema: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['status', 'start', 'stop', 'restart', 'remove', 'logs', 'check', 'update', 'stats'] },
        lines: { type: 'number', description: 'Log lines for action=logs (default 80, max 500) or action=stats (default 2000, max 5000)' },
        force: { type: 'boolean', description: 'For action=update: pull and recreate even if latest tag equals current' },
      },
      required: ['action'],
    },
    risk: 'medium',
    timeoutMs: 180_000,
    async execute(input) {
      try {
        switch (String(input.action)) {
          case 'status': {
            const s = await getStatus(ctx);
            return text(renderStatus(s));
          }
          case 'start': {
            const msg = await startService(ctx);
            await waitHealthy(ctx);
            const s = await getStatus(ctx);
            return text(`${msg}\n${renderStatus(s)}`);
          }
          case 'stop':
            return text(await stopService(ctx));
          case 'restart': {
            const stopMsg = await stopService(ctx).catch(() => '（未在运行，跳过停止）');
            const msg = await startService(ctx);
            await waitHealthy(ctx);
            return text(`${stopMsg}\n${msg}`);
          }
          case 'remove':
            return text(await removeService(ctx));
          case 'logs': {
            const n = Math.min(Math.max(Number(input.lines ?? 80), 1), 500);
            return text(await getLogs(ctx, n));
          }
          case 'check': {
            const items = await checkEnv(ctx);
            return text(items.map((i) => `${i.ok ? '✅' : '❌'} ${i.label}${i.detail ? ` — ${i.detail}` : ''}`).join('\n'));
          }
          case 'stats':
            return text(await getStats(ctx, Math.min(Math.max(Number(input.lines ?? 2000), 100), 5000)));
          case 'update': {
            const msg = await updateImage(ctx, Boolean(input.force));
            await waitHealthy(ctx);
            return text(msg);
          }
          default:
            return text(`未知 action: ${input.action}`, true);
        }
      } catch (err) {
        return text(`操作失败: ${(err as Error).message}`, true);
      }
    },
  }));

  // ---- unified settings menu (simple: status + logs + update check) ----
  // open the panel; optionally jump straight to a tab
  const openPanel = (tab?: 'status' | 'log') => {
    const p = ctx.ui.createPanel(tab ? { payload: { tab } } : undefined);
    ctx.subscriptions.push(p);
    // if the panel was already open, finch:env may not re-fire — nudge it directly
    void p.postMessage({ type: 'show-tab', tab: tab ?? 'status' }).catch(() => undefined);
  };

  ctx.subscriptions.push(ctx.settingsMenu.register({
    async getMenu() {
      const s = await getStatus(ctx);
      const u = await checkUpdate(ctx);
      return [
        { id: 'status', label: s.running ? `运行中 · v${splitImage(readConfig(ctx).image).tag}` : '已停止', disabled: true, iconName: s.running ? 'zap' : 'toggle-left' },
        ...(u.available && u.latest
          ? [{ id: 'update', label: `发现新版本 ${u.latest}`, iconName: 'ext:refresh' as const, hoverText: `当前 ${u.current} → 最新 ${u.latest}\n点击拉取新镜像并重启服务` }]
          : []),
        { id: 'sep1', label: '', separator: true },
        { id: 'logs', label: '查看日志', iconName: 'scroll-text' },
        { id: 'check', label: '环境检查', iconName: 'clipboard-check' },
      ];
    },
    async execute(_menuCtx, itemId) {
      try {
        if (itemId === 'update') {
          const msg = await updateImage(ctx);
          await ctx.ui.showToast({ title: '更新完成', description: msg, variant: 'success' });
        } else if (itemId === 'check') {
          openPanel('status');
        } else if (itemId === 'logs') {
          openPanel('log');
        }
      } catch (err) {
        await ctx.ui.showToast({ title: '操作失败', description: (err as Error).message, variant: 'error' });
      }
    },
  }));

  // ---- composer action: start / stop switch on the input bar ----
  const composerAction = ctx.composerActions.register('halogen-toggle', {
    async getBadge() {
      const s = await getStatus(ctx);
      return s.running ? { text: '运行中', active: true } : '已停止';
    },
    async getIcon() {
      const s = await getStatus(ctx);
      return s.running ? 'zap' : 'ext:power';
    },
    async getMenu() {
      const s = await getStatus(ctx);
      const u = await checkUpdate(ctx);
      return [
        ...(s.running
          ? [{ id: 'stop', label: '停止服务', iconName: 'toggle-left' as const, hoverText: `停止 ${readConfig(ctx).containerName} 容器` }]
          : [{ id: 'start', label: '启动服务', iconName: 'zap' as const, hoverText: '启动模型服务（首次会自动创建容器）' }]),
        { id: 'restart', label: '重启服务', iconName: 'ext:refresh', hoverText: '停止后重新启动' },
        { id: 'sep0', label: '', separator: true },
        { id: 'open', label: '打开控制面板', iconName: 'square-terminal', hoverText: '在右侧面板打开状态 / 日志 / 环境检查' },
        { id: 'logs', label: '查看日志', iconName: 'scroll-text' },
        { id: 'sep1', label: '', separator: true },
        ...(u.available && u.latest
          ? [{ id: 'update', label: `更新到 ${u.latest}`, iconName: 'ext:refresh' as const, hoverText: `当前 ${u.current} → 最新 ${u.latest}\n拉取新镜像并重建容器（会中断服务）` }]
          : [{ id: 'uptodate', label: `已是最新版本${u.current ? ` (${u.current})` : ''}`, iconName: 'check' as const, disabled: true }]),
      ];
    },
    async execute(_actionCtx, itemId) {
      try {
        if (itemId === 'start') await startService(ctx);
        else if (itemId === 'stop') await stopService(ctx);
        else if (itemId === 'restart') { await stopService(ctx).catch(() => undefined); await startService(ctx); }
        else if (itemId === 'open') openPanel('status');
        else if (itemId === 'update') {
          const msg = await updateImage(ctx);
          await ctx.ui.showToast({ title: '更新完成', description: msg, variant: 'success' });
        } else if (itemId === 'logs') {
          openPanel('log');
        }
      } catch (err) {
        await ctx.ui.showToast({ title: '操作失败', description: (err as Error).message, variant: 'error' });
      } finally {
        composerAction.notifyUpdate();
      }
    },
  });
  ctx.subscriptions.push(composerAction);

  // background badge refresh: poll status every 30s, notify on change
  let lastRunning: boolean | undefined;
  const badgeTimer = setInterval(async () => {
    try {
      const s = await getStatus(ctx);
      if (s.running !== lastRunning) { lastRunning = s.running; composerAction.notifyUpdate(); }
    } catch { /* ignore */ }
  }, 30_000);
  ctx.subscriptions.push({ dispose: () => clearInterval(badgeTimer) });

  // ---- panel bridge ----
  ctx.subscriptions.push(ctx.ui.onDidOpenPanel(async (panel) => {
    const streamer = new LogStreamer((msg) => panel.postMessage(msg).catch(() => undefined));
    ctx.subscriptions.push({ dispose: () => streamer.stop() });
    // live update-job mirror: progress survives panel close/reopen
    const updListener = (j: UpdateJob) => {
      void (async () => {
        await panel.postMessage({
          type: 'update-progress',
          text: j.text,
          prog: j.prog,
          done: j.status === 'done',
          error: j.status === 'error',
        }).catch(() => undefined);
        if (j.status !== 'running') {
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) }).catch(() => undefined);
          await panel.postMessage({ type: 'check', items: await checkEnv(ctx) }).catch(() => undefined);
        }
      })();
    };
    updateListeners.add(updListener);
    ctx.subscriptions.push({ dispose: () => updateListeners.delete(updListener) });
    ctx.subscriptions.push(panel.onDidReceiveMessage(async (msg: unknown) => {
      const m = msg as { type?: string; lines?: number; vars?: unknown };
      try {
        if (m.type === 'ready' || m.type === 'refresh') {
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) });
          // on first ready: auto-run env check + update check
          if (m.type === 'ready') {
            await panel.postMessage({ type: 'check', items: await checkEnv(ctx) });
            await panel.postMessage({ type: 'stats', data: await getStatsData(ctx, 2000) });
            // replay in-flight update state
            if (updateJob?.status === 'running') {
              await panel.postMessage({ type: 'update-progress', text: updateJob.text, prog: updateJob.prog });
            }
          }
        } else if (m.type === 'start') {
          const msg2 = await startService(ctx);
          await panel.postMessage({ type: 'event', text: msg2 });
          await waitHealthy(ctx);
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) });
        } else if (m.type === 'stop') {
          const msg2 = await stopService(ctx);
          await panel.postMessage({ type: 'event', text: msg2 });
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) });
        } else if (m.type === 'restart') {
          await stopService(ctx).catch(() => undefined);
          const msg2 = await startService(ctx);
          await panel.postMessage({ type: 'event', text: msg2 });
          await waitHealthy(ctx);
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) });
        } else if (m.type === 'recreate') {
          const msg2 = await recreateService(ctx);
          await panel.postMessage({ type: 'event', text: msg2 });
          await waitHealthy(ctx);
          await panel.postMessage({ type: 'status', data: await getStatus(ctx) });
        } else if (m.type === 'log-start') {
          await streamer.start(ctx, Math.min(Math.max(Number(m.lines ?? 200), 1), 2000));
        } else if (m.type === 'log-stop') {
          streamer.stop();
        } else if (m.type === 'options-get') {
          await panel.postMessage({ type: 'options', data: getOptions() });
        } else if (m.type === 'options-save') {
          const raw = (m.vars ?? {}) as Partial<ServerOptions>;
          const opts: ServerOptions = {
            vision: Boolean(raw.vision),
            qsaForceSel: Boolean(raw.qsaForceSel),
            apiKey: String(raw.apiKey ?? '').trim(),
            thinking: (['default', 'on', 'off'] as const).includes(raw.thinking as 'default') ? raw.thinking! : 'default',
            reasoningEffort: (['', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const).includes(raw.reasoningEffort as 'xhigh') ? raw.reasoningEffort! : '',
            kvSlots: Math.min(Math.max(Math.round(Number(raw.kvSlots) || 4), 1), 64),
            kv8: Boolean(raw.kv8),
            kvPool: [0, 1048576, 2097152].includes(Number(raw.kvPool)) ? Number(raw.kvPool) : 0,
            cacheEntries: Math.min(Math.max(Math.round(Number(raw.cacheEntries) || 0), 0), 256),
            cacheBranches: Number(raw.cacheBranches) === 3 ? 3 : 0,
            port: Math.min(Math.max(Math.round(Number(raw.port) || 8731), 1), 65535),
            lan: Boolean(raw.lan),
            statsApi: Boolean(raw.statsApi),
            statsApiPort: Math.min(Math.max(Math.round(Number(raw.statsApiPort) || 8732), 1), 65535),
          };
          const { apiKey, ...rest } = opts;
          optionsCache = { ...rest, apiKey }; // keep in memory for panel echo; persisted copy excludes it
          await ctx.storage.set('serverOptions', rest);
          if (apiKey) await ctx.secrets.set('halogen.apiKey', apiKey);
          else await ctx.secrets.delete('halogen.apiKey').catch(() => undefined);
          await panel.postMessage({ type: 'options', data: getOptions() });
          await panel.postMessage({ type: 'event', text: '设置已保存，重启容器后生效' });
          syncStatsApi(ctx);
        } else if (m.type === 'logs') {
          await panel.postMessage({ type: 'logs', text: await getLogs(ctx, Math.min(Number(m.lines ?? 200), 500)) });
        } else if (m.type === 'check') {
          await panel.postMessage({ type: 'check', items: await checkEnv(ctx) });
        } else if (m.type === 'stats') {
          await panel.postMessage({ type: 'stats', data: await getStatsData(ctx, Math.min(Math.max(Number(m.lines ?? 2000), 100), 5000)) });
        } else if (m.type === 'update') {
          // progress + final state flow through updListener
          await updateImage(ctx).catch(() => undefined);
        } else if (m.type === 'update-cancel') {
          updateJob?.kill?.();
        } else if (m.type === 'metrics') {
          await panel.postMessage({ type: 'metrics', data: await getLiveMetrics(readConfig(ctx).port) });
        } else if (m.type === 'update-check') {
          await panel.postMessage({ type: 'update', data: await checkUpdate(ctx, true) });
        }
      } catch (err) {
        await panel.postMessage({ type: 'event', text: `错误: ${(err as Error).message}` }).catch(() => undefined);
      }
    }));
    panel.postMessage({ type: 'status', data: await getStatus(ctx) }).catch(() => undefined);
  }));
}

function text(t: string, isError = false) {
  return { content: [{ type: 'text' as const, text: t }], isError };
}

async function waitHealthy(ctx: finch.MiniToolContext): Promise<void> {
  const cfg = readConfig(ctx);
  for (let i = 0; i < 24; i++) {
    const h = await probeHealth(cfg.port);
    if (h.reachable) return;
    await new Promise((r) => setTimeout(r, 5000));
  }
}

export function deactivate(): void {
  if (statsServer) { statsServer.close(); statsServer = undefined; statsServerPort = 0; }
  // disposables cleaned up by Finch
}
