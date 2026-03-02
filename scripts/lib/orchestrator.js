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

const VALID_STATUSES = ['pending', 'in_progress', 'blocked', 'done', 'stale'];
const LOCK_STALE_MS = 30_000; // 30 秒后视 lockfile 为过期

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
 * 标记阶段状态
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

  // 校验 stageId 存在于当前 run 的 stages 中
  {
    const activeCheck = readJSON(path.join(getOrchestratorRoot({ cwd }), 'active-run.json'));
    if (activeCheck && activeCheck.run_id) {
      const runCheck = readJSON(path.join(getOrchestratorRoot({ cwd }), 'runs', activeCheck.run_id, 'run.json'));
      if (runCheck && runCheck.stages && !(stageId in runCheck.stages)) {
        throw new Error(`Unknown stageId "${stageId}". Valid stages: ${Object.keys(runCheck.stages).join(', ')}`);
      }
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
  }

  const run = updateRun({ cwd, patch });

  // 记录事件
  const active = readJSON(path.join(getOrchestratorRoot({ cwd }), 'active-run.json'));
  if (active && active.run_id) {
    appendEvent({
      cwd,
      runId: active.run_id,
      type: 'stage_status_change',
      payload: { stage: stageId, status, note: note || null },
    });
  }

  return run;
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
 * 追加事件到 events.ndjson
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

  fs.appendFileSync(eventsPath, line, 'utf8');
}

/**
 * 尽力型 advisory locking
 * 使用 lockfile（wx 模式）防止并发写入
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

  // 检查是否存在过期的 lockfile
  try {
    const stat = fs.statSync(lockPath);
    const age = Date.now() - stat.mtimeMs;
    if (age > LOCK_STALE_MS) {
      // 过期锁，移除
      fs.unlinkSync(lockPath);
    }
  } catch {
    // lockfile 不存在，正常
  }

  // 尝试获取锁
  let fd;
  try {
    fd = fs.openSync(lockPath, 'wx');
    fs.writeSync(fd, JSON.stringify({ pid: process.pid, timestamp: Date.now() }));
    fs.closeSync(fd);
  } catch (err) {
    if (err.code === 'EEXIST') {
      throw new Error(`Run ${runId} is locked by another process`);
    }
    throw err;
  }

  // 执行函数
  try {
    return fn();
  } finally {
    // 释放锁
    try {
      fs.unlinkSync(lockPath);
    } catch {
      // 忽略删除失败
    }
  }
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

module.exports = {
  getOrchestratorRoot,
  loadStages,
  loadActiveRun,
  initRun,
  updateRun,
  markStage,
  setStageStatus,
  fingerprintFiles,
  appendEvent,
  withRunLock,
};
