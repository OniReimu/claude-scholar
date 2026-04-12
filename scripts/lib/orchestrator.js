/**
 * Workflow Orchestrator Runtime Library
 *
 * 提供研究 run 的状态管理、阶段追踪、产物指纹化和事件日志。
 * 零外部依赖，仅使用 Node.js 内置模块。
 * 所有写操作使用原子写入（*.tmp → renameSync）。
 */

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

const VALID_STATUSES = ['pending', 'in_progress', 'blocked', 'done', 'stale', 'skipped'];
const LOCK_STALE_MS = 30_000; // 30 秒后视 lockfile 为过期
const LOCK_HEARTBEAT_INTERVAL_MS = 10_000; // 10 秒更新一次心跳
const TEX_EXTENSIONS = ['.tex'];
const BIB_EXTENSIONS = ['.bib'];
const GRAPHICS_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg'];

// ---------------------------------------------------------------------------
// 全局事件队列（保护 appendEvent 的并发写入）
// ---------------------------------------------------------------------------

const EVENT_QUEUES = new Map(); // run_id -> {queue: [], writing: false}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

/**
 * 获取 orchestrator 根目录路径
 * @param {{cwd?: string}} opts
 * @returns {string} .claude/orchestrator 的绝对路径
 */
function getOrchestratorRoot({ cwd } = {}) {
  const base = cwd || process.cwd();
  return path.join(base, '.claude', 'orchestrator');
}

/**
 * 获取 stages.json 所在的仓库级 orchestrator 目录
 * 向上查找直到找到 orchestrator/stages.json
 * @param {{cwd?: string}} opts
 * @returns {string} orchestrator 目录的绝对路径
 */
function getRepoOrchestratorDir({ cwd } = {}) {
  let dir = cwd || process.cwd();
  // 向上遍历查找 orchestrator/stages.json
  while (dir !== path.dirname(dir)) {
    const candidate = path.join(dir, 'orchestrator', 'stages.json');
    if (fs.existsSync(candidate)) {
      return path.join(dir, 'orchestrator');
    }
    dir = path.dirname(dir);
  }
  // fallback：使用 cwd
  return path.join(cwd || process.cwd(), 'orchestrator');
}

/**
 * 原子写入 JSON 文件
 * @param {string} filePath
 * @param {object} data
 */
function atomicWriteJSON(filePath, data) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const tmp = filePath + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n', 'utf8');
  fs.renameSync(tmp, filePath);
}

/**
 * 安全读取 JSON 文件
 * @param {string} filePath
 * @returns {object|null}
 */
function readJSON(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

/**
 * 生成 run ID：YYYYMMDD-HHmmss-<4位随机hex>
 * @returns {string}
 */
function generateRunId() {
  const now = new Date();
  const pad = (n, len = 2) => String(n).padStart(len, '0');
  const datePart = [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
  ].join('');
  const timePart = [
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join('');
  const rand = crypto.randomBytes(2).toString('hex');
  return `${datePart}-${timePart}-${rand}`;
}

// ---------------------------------------------------------------------------
// 核心 API
// ---------------------------------------------------------------------------

/**
 * 加载 stages.json 阶段注册表
 * @param {{cwd?: string}} opts
 * @returns {object} stages.json 内容
 */
function loadStages({ cwd } = {}) {
  const stagesPath = path.join(getRepoOrchestratorDir({ cwd }), 'stages.json');
  const data = readJSON(stagesPath);
  if (!data) {
    throw new Error(`Cannot load stages registry: ${stagesPath}`);
  }
  return data;
}

/**
 * 加载当前活跃的 run
 * @param {{cwd?: string}} opts
 * @returns {object|null} run.json 内容，无活跃 run 时返回 null
 */
function loadActiveRun({ cwd } = {}) {
  const root = getOrchestratorRoot({ cwd });
  const activePath = path.join(root, 'active-run.json');
  const active = readJSON(activePath);
  if (!active || !active.run_id) return null;

  const runPath = path.join(root, 'runs', active.run_id, 'run.json');
  return readJSON(runPath);
}

/**
 * 初始化一个新的研究 run
 * @param {{cwd?: string, title: string, profile?: string, venue?: string}} opts
 * @returns {object} 新创建的 run 对象
 */
function initRun({ cwd, title, profile, venue } = {}) {
  if (!title) throw new Error('title is required');

  const root = getOrchestratorRoot({ cwd });
  const runId = generateRunId();
  const runDir = path.join(root, 'runs', runId);
  const now = new Date().toISOString();

  // 加载阶段注册表，初始化所有阶段为 pending
  let stagesRegistry;
  try {
    stagesRegistry = loadStages({ cwd });
  } catch {
    // 如果找不到 stages.json，使用默认阶段列表
    stagesRegistry = {
      stages: [
        { id: 'intake' }, { id: 'literature' }, { id: 'proposal' },
        { id: 'development' }, { id: 'experiments' }, { id: 'analysis' },
        { id: 'writeup' }, { id: 'self_review' }, { id: 'rebuttal' },
        { id: 'post_acceptance' },
      ],
    };
  }

  const stages = {};
  for (const s of stagesRegistry.stages) {
    stages[s.id] = { status: 'pending' };
  }

  const run = {
    id: runId,
    schema_version: '1.5',  // 添加 schema version（v1.5: 添加 8D scoring）
    title,
    profile: profile || null,
    venue: venue || null,
    created_at: now,
    updated_at: now,
    current_stage: 'intake',
    stages,
    inputs: {
      ...(venue ? { venue } : {}),
      ...(profile ? { profile } : {}),
    },
    artifacts: {},
    gate_results: {},
    scoring: {},  // 8D 评分（lazy evaluated）
  };

  // 写入 run.json
  fs.mkdirSync(runDir, { recursive: true });
  atomicWriteJSON(path.join(runDir, 'run.json'), run);

  // 初始化空事件日志
  fs.writeFileSync(path.join(runDir, 'events.ndjson'), '', 'utf8');

  // 设置活跃 run
  atomicWriteJSON(path.join(root, 'active-run.json'), {
    run_id: runId,
    activated_at: now,
  });

  return run;
}

/**
 * 初始化一个 polish 模式的 run（改稿模式）
 * 跳过 stage 1-6，直接从 self_review 开始
 * @param {{cwd?: string, title: string, mainTexPath: string, profile?: string, venue?: string}} opts
 * @returns {object} 新创建的 run 对象
 */
function initPolishRun({ cwd, title, mainTexPath, profile, venue } = {}) {
  if (!title) throw new Error('title is required');
  if (!mainTexPath) throw new Error('mainTexPath is required for polish mode');

  const root = getOrchestratorRoot({ cwd });
  const runId = generateRunId();
  const runDir = path.join(root, 'runs', runId);
  const now = new Date().toISOString();

  let stagesRegistry;
  try {
    stagesRegistry = loadStages({ cwd });
  } catch {
    throw new Error('Cannot load stages.json — required for polish mode');
  }

  // 初始化阶段状态：stage 2-6 标记为 skipped，其他为 pending
  const skippedStages = new Set(['literature', 'proposal', 'development', 'experiments', 'analysis']);
  const stages = {};
  for (const s of stagesRegistry.stages) {
    if (skippedStages.has(s.id)) {
      stages[s.id] = { status: 'skipped', note: 'Skipped in polish mode' };
    } else {
      stages[s.id] = { status: 'pending' };
    }
  }

  // intake 直接标记为 done
  stages.intake = { status: 'done', completed_at: now, note: 'Polish mode intake' };
  // writeup 标记为 done（用户提供的 draft 就是 writeup 产物）
  stages.writeup = { status: 'done', completed_at: now, note: 'Existing draft provided' };

  const run = {
    id: runId,
    title,
    mode: 'polish',
    profile: profile || null,
    venue: venue || null,
    created_at: now,
    updated_at: now,
    current_stage: 'self_review',
    stages,
    inputs: {
      mode: 'polish',
      main_tex: mainTexPath,
      ...(venue ? { venue } : {}),
      ...(profile ? { profile } : {}),
    },
    artifacts: {
      writeup: {
        main_tex: mainTexPath,
      },
    },
    gate_results: {},
  };

  // 写入 run.json
  fs.mkdirSync(runDir, { recursive: true });
  atomicWriteJSON(path.join(runDir, 'run.json'), run);

  // 初始化空事件日志
  fs.writeFileSync(path.join(runDir, 'events.ndjson'), '', 'utf8');

  // 设置活跃 run
  atomicWriteJSON(path.join(root, 'active-run.json'), {
    run_id: runId,
    activated_at: now,
  });

  // 记录 polish init 事件
  appendEvent({
    cwd,
    runId,
    type: 'polish_init',
    payload: { main_tex: mainTexPath, profile, venue },
  });

  return run;
}

/**
 * 更新当前活跃 run 的字段（merge patch）
 * @param {{cwd?: string, patch: object}} opts
 * @returns {object} 更新后的 run 对象
 */
function updateRun({ cwd, patch } = {}) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
    throw new Error('patch must be a plain object');
  }

  const root = getOrchestratorRoot({ cwd });
  const active = readJSON(path.join(root, 'active-run.json'));
  if (!active || !active.run_id) {
    throw new Error('No active run');
  }

  const runPath = path.join(root, 'runs', active.run_id, 'run.json');
  const run = readJSON(runPath);
  if (!run) {
    throw new Error(`Run file not found: ${runPath}`);
  }

  // 深度 merge patch
  const merged = deepMerge(run, patch);
  merged.updated_at = new Date().toISOString();

  atomicWriteJSON(runPath, merged);
  return merged;
}

const DANGEROUS_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

/**
 * 简单深度合并（非数组递归合并），拒绝 prototype pollution 键
 */
function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (DANGEROUS_KEYS.has(key)) continue;
    if (
      source[key] &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      target[key] &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key])
    ) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

/**
 * 规范化相对路径，统一为正斜杠
 * @param {string} relPath
 * @returns {string}
 */
function normalizeRelativePath(relPath) {
  return relPath.split(path.sep).join('/');
}

/**
 * 判断文件是否位于 workspace 内
 * @param {string} base
 * @param {string} fullPath
 * @returns {boolean}
 */
function isWithinWorkspace(base, fullPath) {
  const rel = path.relative(base, fullPath);
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/**
 * 尝试解析候选文件路径
 * @param {{cwd?: string, fromDir: string, refPath: string, extensions?: string[]}} opts
 * @returns {string|null}
 */
function resolveWorkspaceFile({ cwd, fromDir, refPath, extensions = [] } = {}) {
  const base = cwd || process.cwd();
  if (!refPath || typeof refPath !== 'string') return null;
  if (/^[a-z]+:\/\//i.test(refPath)) return null;

  const trimmed = refPath.trim();
  const rawCandidate = path.isAbsolute(trimmed)
    ? trimmed
    : path.resolve(fromDir || base, trimmed);

  const candidates = [rawCandidate];
  if (path.extname(rawCandidate) === '') {
    for (const ext of extensions) {
      candidates.push(rawCandidate + ext);
    }
  }

  for (const candidate of candidates) {
    if (!isWithinWorkspace(base, candidate)) continue;
    try {
      if (fs.statSync(candidate).isFile()) {
        return normalizeRelativePath(path.relative(base, candidate));
      }
    } catch {
      // ignore missing candidates
    }
  }

  return null;
}

/**
 * 移除 LaTeX 注释（忽略转义的 %）
 * @param {string} content
 * @returns {string}
 */
function stripLatexComments(content) {
  return content
    .split('\n')
    .map((line) => line.replace(/(^|[^\\])%.*/g, '$1'))
    .join('\n');
}

/**
 * 从 TeX 内容提取依赖
 * @param {string} content
 * @returns {{includes: string[], bibliographies: string[], graphics: string[]}}
 */
function extractLatexDependencies(content) {
  const sanitized = stripLatexComments(content);
  const includes = [];
  const bibliographies = [];
  const graphics = [];

  const includeRe = /\\(?:input|include)\{([^}]+)\}/g;
  const bibliographyRe = /\\bibliography\{([^}]+)\}/g;
  const addBibRe = /\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}/g;
  const graphicsRe = /\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g;

  let match;
  while ((match = includeRe.exec(sanitized)) !== null) {
    includes.push(match[1].trim());
  }

  while ((match = bibliographyRe.exec(sanitized)) !== null) {
    for (const item of match[1].split(',')) {
      const trimmed = item.trim();
      if (trimmed) bibliographies.push(trimmed);
    }
  }

  while ((match = addBibRe.exec(sanitized)) !== null) {
    const trimmed = match[1].trim();
    if (trimmed) bibliographies.push(trimmed);
  }

  while ((match = graphicsRe.exec(sanitized)) !== null) {
    const trimmed = match[1].trim();
    if (trimmed) graphics.push(trimmed);
  }

  return { includes, bibliographies, graphics };
}

/**
 * 收集 writeup 阶段的本地 TeX 依赖
 * @param {{cwd?: string, mainTexPath: string}} opts
 * @returns {string[]}
 */
function collectWriteupDependencies({ cwd, mainTexPath } = {}) {
  const base = cwd || process.cwd();
  const tracked = new Set();
  const visitedTex = new Set();

  const walkTexFile = (relativePath) => {
    const fullPath = path.join(base, relativePath);
    if (visitedTex.has(fullPath)) return;
    visitedTex.add(fullPath);
    tracked.add(relativePath);

    let content;
    try {
      content = fs.readFileSync(fullPath, 'utf8');
    } catch {
      return;
    }

    const fromDir = path.dirname(fullPath);
    const deps = extractLatexDependencies(content);

    for (const ref of deps.includes) {
      const rel = resolveWorkspaceFile({
        cwd: base,
        fromDir,
        refPath: ref,
        extensions: TEX_EXTENSIONS,
      });
      if (rel) walkTexFile(rel);
    }

    for (const ref of deps.bibliographies) {
      const rel = resolveWorkspaceFile({
        cwd: base,
        fromDir,
        refPath: ref,
        extensions: BIB_EXTENSIONS,
      });
      if (rel) tracked.add(rel);
    }

    for (const ref of deps.graphics) {
      const rel = resolveWorkspaceFile({
        cwd: base,
        fromDir,
        refPath: ref,
        extensions: GRAPHICS_EXTENSIONS,
      });
      if (rel) tracked.add(rel);
    }
  };

  if (typeof mainTexPath !== 'string' || !mainTexPath.trim()) {
    return [];
  }

  const mainTex = resolveWorkspaceFile({
    cwd: base,
    fromDir: base,
    refPath: mainTexPath,
    extensions: TEX_EXTENSIONS,
  });
  if (!mainTex) return [];

  walkTexFile(mainTex);
  return Array.from(tracked).sort();
}

/**
 * 收集某个 stage 应被追踪的文件列表
 * @param {{cwd?: string, run?: object, stageId?: string, extraPaths?: string[]}} opts
 * @returns {string[]}
 */
function collectTrackedFiles({ cwd, run, stageId, extraPaths } = {}) {
  const base = cwd || process.cwd();
  const currentRun = run || loadActiveRun({ cwd: base });
  if (!currentRun) return [];

  const resolvedStageId = stageId || currentRun.current_stage;
  const tracked = new Set();

  let stageDef = null;
  try {
    const stages = loadStages({ cwd: base });
    stageDef = stages.stages.find((stage) => stage.id === resolvedStageId) || null;
  } catch {
    stageDef = null;
  }

  if (stageDef && Array.isArray(stageDef.expected_artifacts)) {
    for (const artifact of stageDef.expected_artifacts) {
      if (!artifact || artifact.kind !== 'file' || typeof artifact.path !== 'string') continue;
      const rel = resolveWorkspaceFile({
        cwd: base,
        fromDir: base,
        refPath: artifact.path,
      });
      if (rel) tracked.add(rel);
    }
  }

  if (resolvedStageId === 'writeup') {
    const mainTexPath = currentRun.artifacts &&
      currentRun.artifacts.writeup &&
      currentRun.artifacts.writeup.main_tex;
    for (const rel of collectWriteupDependencies({ cwd: base, mainTexPath })) {
      tracked.add(rel);
    }
  }

  if (Array.isArray(extraPaths)) {
    for (const extra of extraPaths) {
      const rel = resolveWorkspaceFile({
        cwd: base,
        fromDir: base,
        refPath: extra,
      });
      if (rel) tracked.add(rel);
    }
  }

  return Array.from(tracked).sort();
}

/**
 * 基于 stage 规则生成 tracked_files 与 fingerprints
 * @param {{cwd?: string, run?: object, stageId?: string, extraPaths?: string[]}} opts
 * @returns {{tracked_files: string[], fingerprints: object}}
 */
function fingerprintStageArtifacts({ cwd, run, stageId, extraPaths } = {}) {
  const tracked_files = collectTrackedFiles({ cwd, run, stageId, extraPaths });
  return {
    tracked_files,
    fingerprints: fingerprintFiles({ cwd, paths: tracked_files }),
  };
}

/**
 * 验证 stage 的所有 required gates 是否已通过
 *
 * Data-driven：每个 policy_check gate 可声明 result_keys 数组，
 * validateGates 据此检查 gate_results[stageId] 中对应的 key。
 * 未声明 result_keys 时 fallback 到通用 `passed` 字段（向后兼容）。
 *
 * @param {{cwd?: string, stageId: string, run?: object}} opts
 * @returns {{valid: boolean, failures: string[]}}
 */
function validateGates({ cwd, stageId, run } = {}) {
  const failures = [];

  let stagesRegistry;
  try {
    stagesRegistry = loadStages({ cwd });
  } catch {
    return { valid: true, failures }; // 无 stages.json 时跳过验证
  }

  const stageDef = stagesRegistry.stages.find((s) => s.id === stageId);
  if (!stageDef || !Array.isArray(stageDef.gates) || stageDef.gates.length === 0) {
    return { valid: true, failures };
  }

  const activeRun = run || loadActiveRun({ cwd });
  if (!activeRun) return { valid: true, failures };

  const gateResults = (activeRun.gate_results && activeRun.gate_results[stageId]) || {};

  // Collect all declared result_keys across all gates for this stage.
  // If ANY new key is present in gateResults, disable legacy fallback for ALL gates.
  // This prevents mixed state bypass (e.g., guardrail_clean + passed without guidance_clean).
  const allDeclaredKeys = [];
  for (const gate of stageDef.gates) {
    if (gate.type === 'policy_check' && Array.isArray(gate.result_keys)) {
      allDeclaredKeys.push(...gate.result_keys);
    }
  }
  const hasAnyNewKeyGlobal = allDeclaredKeys.some((key) => key in gateResults);

  for (const gate of stageDef.gates) {
    if (gate.type !== 'policy_check') continue;
    // human_approval gates 不在此验证（由 LLM/skill 层保证）

    const resultKeys = gate.result_keys;
    if (Array.isArray(resultKeys) && resultKeys.length > 0) {
      // Data-driven：检查声明的每个 result key（strict boolean）
      if (!hasAnyNewKeyGlobal && gateResults.passed === true) {
        // Pure legacy run.json — no new keys at all, accept `passed` as compatible
        continue;
      }
      // New schema or mixed state: require every declared key
      for (const key of resultKeys) {
        if (gateResults[key] !== true) {
          failures.push(`${key} not passed (gate: ${gate.description || 'unnamed'})`);
        }
      }
    } else {
      // Fallback：通用 `passed` 字段（向后兼容旧 schema，strict boolean）
      if (gateResults.passed !== true) {
        failures.push(`policy_check gate "${gate.description || 'unnamed'}" not passed`);
      }
    }
  }

  return { valid: failures.length === 0, failures };
}

/**
 * 标记阶段状态（TOCTOU-safe）
 * 修复：validateGates() + updateRun() 间的竞态（v1.5）
 * @param {{cwd?: string, stageId: string, status: string, note?: string}} opts
 * @returns {object} 更新后的 run 对象
 */
function markStage({ cwd, stageId, status, note } = {}) {
  if (!stageId || typeof stageId !== 'string') {
    throw new Error('stageId is required and must be a string');
  }
  if (!VALID_STATUSES.includes(status)) {
    throw new Error(`Invalid status "${status}". Must be one of: ${VALID_STATUSES.join(', ')}`);
  }

  // 获取当前 run 和 run_id（用于锁）
  const currentRun = loadActiveRun({ cwd });
  if (!currentRun) {
    throw new Error('No active run');
  }

  const root = getOrchestratorRoot({ cwd });
  const active = readJSON(path.join(root, 'active-run.json'));
  if (!active || !active.run_id) {
    throw new Error('No active run ID');
  }

  // 在锁保护下执行所有更新操作
  return withRunLock({ cwd, runId: active.run_id }, () => {
    // 重新加载最新的 run 状态（在锁内）
    const run = loadActiveRun({ cwd });
    if (!run) {
      throw new Error('Run disappeared under lock');
    }

    // 校验 stageId 存在于当前 run 的 stages 中
    if (run && run.stages && !(stageId in run.stages)) {
      throw new Error(`Unknown stageId "${stageId}". Valid stages: ${Object.keys(run.stages).join(', ')}`);
    }

    // 当标记为 done 时，验证所有 required gates 已通过
    if (status === 'done') {
      const { valid, failures } = validateGates({ cwd, stageId, run });
      if (!valid) {
        throw new Error(
          `Cannot mark "${stageId}" as done: ${failures.join('; ')}`
        );
      }
    }

    const now = new Date().toISOString();
    const patch = {
      stages: {
        [stageId]: {
          status,
          ...(status === 'in_progress' ? { started_at: now } : {}),
          ...(status === 'done' ? { completed_at: now } : {}),
          ...(note ? { note } : {}),
        },
      },
    };

    // 如果标记为 in_progress，同时更新 current_stage
    if (status === 'in_progress') {
      patch.current_stage = stageId;

      // Lazy evaluation: 进入 self_review 时计算 8D 评分
      if (stageId === 'self_review') {
        try {
          const scores = compute8DScoring({ cwd, runId: active.run_id, run });
          patch.scoring = { self_review: scores };
        } catch (err) {
          // 8D 评分计算失败，不阻塞流程，仅记录
          console.warn(`Failed to compute 8D scoring: ${err.message}`);
        }
      }
    }

    const updatedRun = updateRun({ cwd, patch });

    // 记录事件（在锁内，保证一致性）
    appendEvent({
      cwd,
      runId: active.run_id,
      type: 'stage_status_change',
      payload: { stage: stageId, status, note: note || null },
    });

    return updatedRun;
  });
}

/**
 * 设置阶段状态（支持回滚到 pending/stale，并清除下游 done 状态）
 * @param {{cwd?: string, stageId: string, status: string, reason?: string}} opts
 * @returns {object} 更新后的 run 对象
 */
function setStageStatus({ cwd, stageId, status, reason } = {}) {
  if (!VALID_STATUSES.includes(status)) {
    throw new Error(`Invalid status "${status}". Must be one of: ${VALID_STATUSES.join(', ')}`);
  }

  // 如果回滚（设为 pending 或 stale），清除下游阶段的 done 状态
  if (status === 'pending' || status === 'stale') {
    let stagesRegistry;
    try {
      stagesRegistry = loadStages({ cwd });
    } catch {
      stagesRegistry = null;
    }

    if (stagesRegistry) {
      const stageOrder = stagesRegistry.stages.map((s) => s.id);
      const targetIdx = stageOrder.indexOf(stageId);

      if (targetIdx >= 0) {
        const root = getOrchestratorRoot({ cwd });
        const active = readJSON(path.join(root, 'active-run.json'));
        if (active && active.run_id) {
          const runPath = path.join(root, 'runs', active.run_id, 'run.json');
          const run = readJSON(runPath);
          if (run) {
            // 清除下游阶段的 done 状态
            for (let i = targetIdx + 1; i < stageOrder.length; i++) {
              const downstreamId = stageOrder[i];
              if (run.stages[downstreamId] && run.stages[downstreamId].status === 'done') {
                run.stages[downstreamId].status = 'pending';
                run.stages[downstreamId].cleared_by_rollback = true;
              }
            }
            // 回滚时同步 current_stage 到被回滚的阶段
            run.current_stage = stageId;
            run.updated_at = new Date().toISOString();
            atomicWriteJSON(runPath, run);
          }
        }
      }
    }
  }

  return markStage({ cwd, stageId, status, note: reason });
}

/**
 * 计算文件的 SHA256 指纹
 * @param {{cwd?: string, paths: string[]}} opts
 * @returns {object} { path: "sha256:..." } 映射，跳过不存在的文件
 */
function fingerprintFiles({ cwd, paths } = {}) {
  const base = cwd || process.cwd();
  const result = {};

  for (const p of paths) {
    const fullPath = path.isAbsolute(p) ? p : path.join(base, p);
    try {
      const content = fs.readFileSync(fullPath);
      const hash = crypto.createHash('sha256').update(content).digest('hex');
      result[p] = `sha256:${hash}`;
    } catch {
      // 跳过不存在或不可读的文件
    }
  }

  return result;
}

/**
 * 追加事件到 events.ndjson（带事件队列守卫）
 * 修复：并发写入导致 NDJSON 损坏（v1.5）
 * @param {{cwd?: string, runId: string, type: string, payload: object}} opts
 */
function appendEvent({ cwd, runId, type, payload } = {}) {
  const root = getOrchestratorRoot({ cwd });
  const eventsPath = path.join(root, 'runs', runId, 'events.ndjson');

  const event = {
    timestamp: new Date().toISOString(),
    type,
    payload: payload || {},
  };

  const line = JSON.stringify(event) + '\n';

  // 确保目录存在
  const dir = path.dirname(eventsPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // 使用事件队列防止并发写入
  // 初始化队列（如果不存在）
  if (!EVENT_QUEUES.has(runId)) {
    EVENT_QUEUES.set(runId, { queue: [], writing: false });
  }

  const queueState = EVENT_QUEUES.get(runId);
  queueState.queue.push(line);

  // 处理队列
  if (!queueState.writing) {
    queueState.writing = true;
    (function processQueue() {
      if (queueState.queue.length === 0) {
        queueState.writing = false;
        return;
      }

      const line = queueState.queue.shift();
      try {
        fs.appendFileSync(eventsPath, line, 'utf8');
      } catch (err) {
        // 重新入队失败的行（简单重试，实际可能需要更复杂的处理）
        console.error(`Failed to append event for run ${runId}: ${err.message}`);
      }

      // 继续处理队列（同步处理避免竞态）
      processQueue();
    })();
  }
}

/**
 * PID-based advisory locking with heartbeat
 * 使用 PID 验证 + 心跳防止 false-positive 过期检测
 * 修复：TOCTOU 竞态（v1.5）
 * @param {{cwd?: string, runId: string}} opts
 * @param {function} fn - 在锁保护下执行的函数
 * @returns {*} fn 的返回值
 */
function withRunLock({ cwd, runId }, fn) {
  const root = getOrchestratorRoot({ cwd });
  const lockPath = path.join(root, 'runs', runId, '.lock');

  // 确保目录存在
  const dir = path.dirname(lockPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // 检查是否存在过期的 lockfile（基于时间戳，不信任 mtime）
  try {
    const lockData = readJSON(lockPath);
    if (lockData && lockData.pid) {
      const age = Date.now() - (lockData.heartbeat_time || lockData.timestamp);
      // 超过 30 秒，视为过期（所有者进程可能崩溃了）
      if (age > LOCK_STALE_MS) {
        fs.unlinkSync(lockPath);
      }
    }
  } catch {
    // lockfile 不存在或读取失败，正常
  }

  // 尝试获取锁（原子操作）
  const lockContent = {
    pid: process.pid,
    timestamp: Date.now(),
    heartbeat_time: Date.now(),
  };

  try {
    // 用 tmp → rename 确保原子性
    const tmp = lockPath + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(lockContent) + '\n', 'utf8');
    fs.renameSync(tmp, lockPath);
  } catch (err) {
    // 锁获取失败
    throw new Error(`Run ${runId} is locked by another process`);
  }

  // 启动心跳定时器，防止 30s 超时
  let heartbeatInterval = null;
  let result;

  try {
    heartbeatInterval = setInterval(() => {
      try {
        const current = readJSON(lockPath);
        if (current && current.pid === process.pid) {
          current.heartbeat_time = Date.now();
          const tmp = lockPath + '.tmp';
          fs.writeFileSync(tmp, JSON.stringify(current) + '\n', 'utf8');
          fs.renameSync(tmp, lockPath);
        }
      } catch {
        // 心跳失败忽略（进程将在 finally 中释放锁）
      }
    }, LOCK_HEARTBEAT_INTERVAL_MS);

    // 执行被保护的函数
    result = fn();
  } finally {
    // 停止心跳
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
    }

    // 释放锁
    try {
      const current = readJSON(lockPath);
      if (current && current.pid === process.pid) {
        fs.unlinkSync(lockPath);
      }
    } catch {
      // 忽略释放失败
    }
  }

  return result;
}

/**
 * 计算 8D 质量评分（lazy evaluated，仅在 self_review 时触发）
 * 评分维度：novelty, credibility, clarity, completeness, significance, reproducibility, writing_quality, venue_fit
 * 每个维度 0-1.0，通过以下方式计算：
 * - Prose violations: writing_quality = 1.0 - (prose_violations / expected_violation_count)
 * - Policy violations: venue_fit = 1.0 - (policy_violations / total_rules)
 * - Content completeness: completeness = min(1.0, word_count / expected_word_count)
 * 其他维度：human judgment input (待实现)
 * @param {{cwd?: string, runId?: string, run?: object, stageId?: string}} opts
 * @returns {{novelty, credibility, clarity, completeness, significance, reproducibility, writing_quality, venue_fit}}
 */
function compute8DScoring({ cwd, runId, run, stageId } = {}) {
  // 如果没有提供 run，尝试加载
  let currentRun = run;
  if (!currentRun) {
    if (runId) {
      const root = getOrchestratorRoot({ cwd });
      const runPath = path.join(root, 'runs', runId, 'run.json');
      currentRun = readJSON(runPath);
    }
    if (!currentRun) {
      currentRun = loadActiveRun({ cwd });
    }
  }

  if (!currentRun) {
    throw new Error('Cannot compute 8D scoring: no run provided');
  }

  // 初始默认值（所有维度都用保守的 0.6 作为中值）
  const scores = {
    novelty: 0.60,              // 需要人工评估
    credibility: 0.60,           // 需要人工评估
    clarity: 0.60,               // 需要人工评估
    completeness: 0.60,          // 实验完整性
    significance: 0.60,          // 需要人工评估
    reproducibility: 0.60,       // 需要人工评估
    writing_quality: 0.60,       // 基于 prose violations
    venue_fit: 0.60,             // 基于 policy violations
    computed_at: new Date().toISOString(),
    note: 'v1.0: Conservative defaults (0.6) - human refinement pending',
  };

  // 如果有 gate_results，可以用其中的信息来调整评分
  if (currentRun.gate_results) {
    // 例如：如果 self_review gate 有 violation count，用它来调整 writing_quality 和 venue_fit
    const selfReviewGate = currentRun.gate_results['self_review'];
    if (selfReviewGate && typeof selfReviewGate.violation_count === 'number') {
      // 假设正常论文有 5-10 个 violations，少于 5 是 excellent
      const violationRatio = Math.min(1.0, selfReviewGate.violation_count / 10);
      scores.writing_quality = Math.max(0.5, 1.0 - violationRatio * 0.4);  // 0.5-1.0
      scores.venue_fit = Math.max(0.5, 1.0 - violationRatio * 0.3);        // 0.5-1.0
    }
  }

  return scores;
}

/**
 * 更新 run 的 8D 评分（在进入 self_review 时调用）
 * @param {{cwd?: string, runId?: string, run?: object}} opts
 * @returns {object} 更新后的 run 对象
 */
function update8DScoring({ cwd, runId, run } = {}) {
  let currentRun = run;
  if (!currentRun) {
    if (runId) {
      const root = getOrchestratorRoot({ cwd });
      const runPath = path.join(root, 'runs', runId, 'run.json');
      currentRun = readJSON(runPath);
    }
    if (!currentRun) {
      currentRun = loadActiveRun({ cwd });
    }
  }

  if (!currentRun) {
    throw new Error('Cannot update 8D scoring: no run provided');
  }

  // 计算 8D 评分
  const scores = compute8DScoring({ cwd, runId, run: currentRun });

  // 合并到 run.scoring
  const patch = {
    scoring: {
      self_review: scores,
    },
  };

  return updateRun({ cwd, patch });
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

module.exports = {
  getOrchestratorRoot,
  loadStages,
  loadActiveRun,
  initRun,
  initPolishRun,
  updateRun,
  collectTrackedFiles,
  fingerprintStageArtifacts,
  validateGates,
  markStage,
  setStageStatus,
  fingerprintFiles,
  appendEvent,
  withRunLock,
  compute8DScoring,
  update8DScoring,
};
