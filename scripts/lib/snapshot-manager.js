/**
 * Multi-version Snapshot Manager
 *
 * 为论文保存 hash-indexed immutable snapshots，支持回滚和对比。
 * 用于 Phase 2-3 的版本控制和 rollback safety。
 *
 * 架构：
 * - snapshotPaper(): 对 main.tex 和依赖项进行 hash 和压缩快照
 * - listSnapshots(): 列出所有快照
 * - restoreSnapshot(): 恢复到某个快照
 * - compareSnapshots(): 对比两个快照的差异
 *
 * 存储：.claude/snapshots/{stage}/{hash}/
 */

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

// ---------------------------------------------------------------------------
// 常量和配置
// ---------------------------------------------------------------------------

const SNAPSHOT_RETENTION_DAYS = 30;  // 保留 30 天的快照
const MAX_SNAPSHOTS_PER_STAGE = 50;  // 每个阶段最多 50 个快照

// ---------------------------------------------------------------------------
// 哈希和压缩函数
// ---------------------------------------------------------------------------

/**
 * 计算文件的 SHA256 哈希
 * @param {string} filePath
 * @returns {string} hex 格式的哈希
 */
function hashFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return crypto.createHash('sha256').update(content).digest('hex').substring(0, 12);
  } catch {
    return null;
  }
}

/**
 * 计算文件集合的合并哈希
 * @param {string[]} filePaths - 要包含的文件路径数组
 * @returns {string} 合并后的哈希
 */
function hashFiles(filePaths) {
  const hashes = [];
  for (const fp of filePaths) {
    const h = hashFile(fp);
    if (h) hashes.push(h);
  }
  const combined = hashes.join('|');
  return crypto.createHash('sha256').update(combined).digest('hex').substring(0, 12);
}

/**
 * 收集 LaTeX 文件的依赖项（.tex, .bib, .sty）
 * @param {string} mainTexPath - main.tex 的路径
 * @returns {string[]} 依赖文件列表
 */
function collectDependencies(mainTexPath) {
  if (!fs.existsSync(mainTexPath)) {
    return [];
  }

  const baseDir = path.dirname(mainTexPath);
  const dependencies = new Set([mainTexPath]);

  try {
    const content = fs.readFileSync(mainTexPath, 'utf8');

    // 查找 \input{...}、\include{...}、\bibliography{...}、\usepackage[path]{...}
    const patterns = [
      /\\(?:input|include)\{([^}]+)\}/g,
      /\\bibliography\{([^}]+)\}/g,
      /\\usepackage\[path\]\{([^}]+)\}/g,
    ];

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        let depPath = match[1];
        if (!depPath.endsWith('.tex') && !depPath.endsWith('.bib')) {
          depPath += '.tex';  // 默认添加 .tex 扩展名
        }
        const fullPath = path.join(baseDir, depPath);
        if (fs.existsSync(fullPath)) {
          dependencies.add(fullPath);
        }
      }
    }
  } catch {
    // 忽略读取错误
  }

  return Array.from(dependencies);
}

// ---------------------------------------------------------------------------
// 快照操作
// ---------------------------------------------------------------------------

/**
 * 创建快照
 * @param {{cwd: string, mainTexPath: string, stageId: string, metadata?: {}}} opts
 * @returns {{snapshotId: string, contentHash: string, timestamp: string, metadata: {}}}
 */
function createSnapshot({ cwd, mainTexPath, stageId, metadata } = {}) {
  if (!fs.existsSync(mainTexPath)) {
    throw new Error(`main.tex not found: ${mainTexPath}`);
  }

  const baseDir = path.dirname(mainTexPath);
  const snapshotDir = path.join(cwd, '.claude', 'snapshots');

  // 收集依赖并计算合并哈希
  const dependencies = collectDependencies(mainTexPath);
  const contentHash = hashFiles(dependencies);

  if (!contentHash) {
    throw new Error('Failed to hash main.tex');
  }

  // 生成快照 ID（stage + timestamp + hash）
  const now = new Date();
  const timestamp = now.toISOString();
  const dateStr = now.toISOString().split('T')[0].replace(/-/g, '');
  const timeStr = now.toISOString().split('T')[1].substring(0, 6).replace(/:/g, '');
  const snapshotId = `${stageId}-${dateStr}T${timeStr}-${contentHash}`;

  const stagePath = path.join(snapshotDir, stageId);
  const hashPath = path.join(stagePath, contentHash);

  // 确保目录存在
  fs.mkdirSync(hashPath, { recursive: true });

  // 保存快照清单
  const manifest = {
    snapshotId,
    contentHash,
    stageId,
    timestamp,
    mainTexPath: path.relative(baseDir, mainTexPath),
    dependencies: dependencies.map(d => path.relative(baseDir, d)),
    metadata: metadata || {},
  };

  fs.writeFileSync(path.join(hashPath, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');

  // 保存依赖文件（原始内容）
  for (const depPath of dependencies) {
    const relPath = path.relative(baseDir, depPath);
    const savePath = path.join(hashPath, relPath);
    const saveDir = path.dirname(savePath);

    fs.mkdirSync(saveDir, { recursive: true });
    fs.copyFileSync(depPath, savePath);
  }

  // 清理旧快照（超过保留天数或超过最大数量）
  cleanupOldSnapshots(stagePath, SNAPSHOT_RETENTION_DAYS, MAX_SNAPSHOTS_PER_STAGE);

  return manifest;
}

/**
 * 列出某个阶段的所有快照
 * @param {{cwd: string, stageId: string}} opts
 * @returns {Array} 快照列表（按时间倒序）
 */
function listSnapshots({ cwd, stageId } = {}) {
  const stagePath = path.join(cwd, '.claude', 'snapshots', stageId);

  if (!fs.existsSync(stagePath)) {
    return [];
  }

  const snapshots = [];
  const hashDirs = fs.readdirSync(stagePath);

  for (const hashDir of hashDirs) {
    const hashPath = path.join(stagePath, hashDir);
    const manifestPath = path.join(hashPath, 'manifest.json');

    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
      snapshots.push(manifest);
    } catch {
      // 忽略无效的快照
    }
  }

  // 按时间倒序排序
  snapshots.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  return snapshots;
}

/**
 * 恢复快照
 * @param {{cwd: string, baseDir: string, snapshotId: string}} opts
 * @returns {object} 恢复的快照清单
 */
function restoreSnapshot({ cwd, baseDir, snapshotId } = {}) {
  const snapshotsPath = path.join(cwd, '.claude', 'snapshots');
  let snapshotPath = null;

  // 查找快照
  for (const stage of fs.readdirSync(snapshotsPath)) {
    for (const hash of fs.readdirSync(path.join(snapshotsPath, stage))) {
      const hPath = path.join(snapshotsPath, stage, hash);
      const manifest = JSON.parse(fs.readFileSync(path.join(hPath, 'manifest.json'), 'utf8'));
      if (manifest.snapshotId === snapshotId) {
        snapshotPath = hPath;
        break;
      }
    }
    if (snapshotPath) break;
  }

  if (!snapshotPath) {
    throw new Error(`Snapshot not found: ${snapshotId}`);
  }

  // 恢复文件
  const manifest = JSON.parse(fs.readFileSync(path.join(snapshotPath, 'manifest.json'), 'utf8'));

  for (const relPath of manifest.dependencies) {
    const srcPath = path.join(snapshotPath, relPath);
    const dstPath = path.join(baseDir, relPath);

    if (fs.existsSync(srcPath)) {
      const dstDir = path.dirname(dstPath);
      fs.mkdirSync(dstDir, { recursive: true });
      fs.copyFileSync(srcPath, dstPath);
    }
  }

  return manifest;
}

/**
 * 比较两个快照
 * @param {{snapshot1: object, snapshot2: object}} opts
 * @returns {object} 差异报告
 */
function compareSnapshots({ snapshot1, snapshot2 } = {}) {
  const files1 = new Set(snapshot1.dependencies || []);
  const files2 = new Set(snapshot2.dependencies || []);

  const added = [...files2].filter(f => !files1.has(f));
  const removed = [...files1].filter(f => !files2.has(f));
  const common = [...files1].filter(f => files2.has(f));

  return {
    snapshot1Id: snapshot1.snapshotId,
    snapshot2Id: snapshot2.snapshotId,
    added,
    removed,
    common,
    filesChanged: added.length + removed.length > 0,
    contentHashChanged: snapshot1.contentHash !== snapshot2.contentHash,
  };
}

/**
 * 清理旧快照
 * @param {string} stagePath
 * @param {number} retentionDays
 * @param {number} maxSnapshots
 */
function cleanupOldSnapshots(stagePath, retentionDays, maxSnapshots) {
  if (!fs.existsSync(stagePath)) return;

  const snapshots = listSnapshots({ cwd: path.dirname(stagePath), stageId: path.basename(stagePath) });

  const now = Date.now();
  const cutoffTime = now - retentionDays * 24 * 60 * 60 * 1000;

  // 标记要删除的快照
  let deleteCount = 0;
  for (let i = snapshots.length; i >= 0; i--) {
    if (i >= maxSnapshots) {
      deleteCount = snapshots.length - maxSnapshots;
      break;
    }
    const snapshotTime = new Date(snapshots[i].timestamp).getTime();
    if (snapshotTime < cutoffTime) {
      deleteCount = snapshots.length - i;
      break;
    }
  }

  // 删除旧快照（从最老的开始）
  for (let i = 0; i < deleteCount && i < snapshots.length; i++) {
    const hashPath = path.join(stagePath, snapshots[i].contentHash);
    try {
      fs.rmSync(hashPath, { recursive: true, force: true });
    } catch {
      // 忽略删除失败
    }
  }
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

module.exports = {
  createSnapshot,
  listSnapshots,
  restoreSnapshot,
  compareSnapshots,
  collectDependencies,
  hashFile,
  hashFiles,
};
