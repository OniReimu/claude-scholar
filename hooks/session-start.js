#!/usr/bin/env node
/**
 * SessionStart Hook: Display project status (cross-platform version)
 *
 * Event: SessionStart
 * Function: Display project status, Git info, todos, plugins, and commands at session start
 */

const path = require('path');
const os = require('os');
const fs = require('fs');

// Import shared utility library
const common = require('./hook-common');

// Import package manager detection
const { getPackageManager, getSelectionPrompt } = require('../scripts/lib/package-manager');

// Read stdin input
let input = {};
try {
  const stdinData = require('fs').readFileSync(0, 'utf8');
  if (stdinData.trim()) {
    input = JSON.parse(stdinData);
  }
} catch {
  // Use default empty object
}

const cwd = input.cwd || process.cwd();
const projectName = path.basename(cwd);
const homeDir = os.homedir();

// Build output
let output = '';

// Session start info
output += `🚀 ${projectName} Session started\n`;
output += `▸ Time: ${common.formatDateTime()}\n`;
output += `▸ Directory: ${cwd}\n\n`;

// Git status
const gitInfo = common.getGitInfo(cwd);

if (gitInfo.is_repo) {
  output += `▸ Git branch: ${gitInfo.branch}\n\n`;

  if (gitInfo.has_changes) {
    output += `⚠️  Uncommitted changes (${gitInfo.changes_count} files):\n`;

    // Show change list (up to 10)
    const statusIcons = {
      'M': '📝',  // Modified
      'A': '➕',  // Added
      'D': '❌',  // Deleted
      'R': '🔄',  // Renamed
      '??': '❓'  // Untracked
    };

    for (let i = 0; i < Math.min(gitInfo.changes.length, 10); i++) {
      const change = gitInfo.changes[i];
      const status = change.substring(0, 2).trim();
      const file = change.substring(3).trim();

      const icon = statusIcons[status] || '•';
      output += `  ${icon} ${file}\n`;
    }

    if (gitInfo.changes_count > 10) {
      output += `  ... (${gitInfo.changes_count - 10} more files)\n`;
    }
  } else {
    output += `✅ Working directory clean\n`;
  }
  output += '\n';
} else {
  output += `▸ Git: Not a repository\n\n`;
}

// Package manager detection
try {
  const pm = getPackageManager();
  output += `📦 Package manager: ${pm.name} (${pm.source})\n`;

  // If detected via fallback, suggest setup
  if (pm.source === 'fallback' || pm.source === 'default') {
    output += `💡 Run /setup-pm to configure preferred package manager\n`;
  }
} catch (err) {
  // Package manager detection failed, silently ignore
}

output += '\n';

// Orchestrator: Active Run Status
try {
  const orchestrator = require('../scripts/lib/orchestrator');
  const activeRun = orchestrator.loadActiveRun({ cwd });
  if (activeRun) {
    let stages;
    try { stages = orchestrator.loadStages({ cwd }); } catch { stages = null; }

    // 指纹比对：检测 done 阶段的 artifact 是否被修改/删除，自动标记 stale
    // 按 stage order 排序，从最上游开始检测（setStageStatus 会清除下游）
    const stageOrder = stages ? stages.stages.map(s => s.id) : [];
    const doneStages = Object.entries(activeRun.stages || {})
      .filter(([, v]) => v.status === 'done')
      .map(([k]) => k);
    if (stageOrder.length > 0) {
      doneStages.sort((a, b) => stageOrder.indexOf(a) - stageOrder.indexOf(b));
    }
    const newlyStale = [];
    for (const stageId of doneStages) {
      // 重新加载 run：前一轮 setStageStatus 可能已清除此 stage 为 pending
      const currentRun = newlyStale.length > 0 ? orchestrator.loadActiveRun({ cwd }) : activeRun;
      if (!currentRun || (currentRun.stages[stageId] || {}).status !== 'done') continue;
      const fingerprints = (currentRun.artifacts && currentRun.artifacts[stageId] && currentRun.artifacts[stageId].fingerprints) || {};
      const filePaths = Object.keys(fingerprints);
      if (filePaths.length === 0) continue;
      const currentHashes = orchestrator.fingerprintFiles({ cwd, paths: filePaths });
      for (const fp of filePaths) {
        // 文件被删除（undefined）或内容变更：都标为 stale
        if (!currentHashes[fp] || currentHashes[fp] !== fingerprints[fp]) {
          const reason = !currentHashes[fp] ? `Artifact missing: ${fp}` : `Artifact changed: ${fp}`;
          try {
            orchestrator.setStageStatus({ cwd, stageId, status: 'stale', reason });
            newlyStale.push(stageId);
          } catch { /* ignore write errors in hook */ }
          break; // 一个 mismatch 就够了
        }
      }
    }
    // 重新加载 run state（可能被 setStageStatus 修改过）
    const freshRun = newlyStale.length > 0 ? orchestrator.loadActiveRun({ cwd }) || activeRun : activeRun;

    const stageLabel = stages
      ? (stages.stages.find(s => s.id === freshRun.current_stage) || {}).label || freshRun.current_stage
      : freshRun.current_stage;
    const stageStatus = (freshRun.stages[freshRun.current_stage] || {}).status || 'unknown';
    const nextStage = stages
      ? (stages.stages.find(s => s.id === freshRun.current_stage) || {}).next_stage || null
      : null;
    output += `🔬 Active Run: ${freshRun.id} — ${freshRun.title}\n`;
    output += `  ▸ Stage: ${stageLabel} [${stageStatus}]\n`;
    if (freshRun.venue) output += `  ▸ Venue: ${freshRun.venue}\n`;
    if (nextStage) output += `  ▸ Next: ${nextStage}\n`;
    // 显示所有 stale 阶段
    const staleStages = Object.entries(freshRun.stages || {})
      .filter(([, v]) => v.status === 'stale')
      .map(([k]) => k);
    if (staleStages.length > 0) {
      output += `  ⚠️  Stale stages: ${staleStages.join(', ')}`;
      if (newlyStale.length > 0) output += ` (${newlyStale.length} detected this session)`;
      output += '\n';
    }
    output += '\n';
  }
} catch {
  // orchestrator 不可用时静默忽略
}

// Todos
output += `📋 Todos:\n`;
const todoInfo = common.getTodoInfo(cwd);

if (todoInfo.found) {
  output += `  - ${todoInfo.pending} pending / ${todoInfo.done} completed\n`;

  // Show top 5 pending items
  if (fs.existsSync(todoInfo.path)) {
    try {
      const content = fs.readFileSync(todoInfo.path, 'utf8');
      const pendingItems = content.match(/^[\-\*] \[ \].+$/gm) || [];

      if (pendingItems.length > 0) {
        output += `\n  Recent todos:\n`;
        for (let i = 0; i < Math.min(5, pendingItems.length); i++) {
          const item = pendingItems[i].replace(/^[\-\*] \[ \]\s*/, '').substring(0, 60);
          output += `  - ${item}\n`;
        }
      }
    } catch {
      // Ignore errors
    }
  }
} else {
  output += `  No todo file found (TODO.md, docs/todo.md etc)\n`;
}

output += '\n';

// Enabled plugins
output += `🔌 Enabled plugins:\n`;
const enabledPlugins = common.getEnabledPlugins(homeDir);

if (enabledPlugins.length > 0) {
  for (const plugin of enabledPlugins) {
    output += `  - ${plugin.name}\n`;
  }
} else {
  output += `  None\n`;
}

output += '\n';

// Available commands
output += `💡 Available commands:\n`;
const availableCommands = common.getAvailableCommands(homeDir);

if (availableCommands.length > 0) {
  for (const cmd of availableCommands.slice(0, 20)) {
    const description = common.getCommandDescription(cmd.path) || `${cmd.plugin} command`;
    const truncatedDesc = description.length > 40 ? description.substring(0, 40) + '...' : description;
    output += `  /${cmd.name.padEnd(20)} ${truncatedDesc}\n`;
  }

  if (availableCommands.length > 20) {
    output += `  ... (${availableCommands.length - 20} more commands)\n`;
  }
} else {
  output += `  No commands found\n`;
}

// Output JSON
const result = {
  continue: true,
  systemMessage: output
};

console.log(JSON.stringify(result));

process.exit(0);
