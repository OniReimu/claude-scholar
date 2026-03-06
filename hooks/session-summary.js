#!/usr/bin/env node
/**
 * SessionEnd Hook: Work Log + Smart Suggestions (Cross-platform)
 *
 * Event: SessionEnd
 * Purpose: Create work log, record changes and generate smart suggestions
 */

const path = require('path');
const fs = require('fs');
const os = require('os');
const common = require('./hook-common');

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
const sessionId = input.session_id || 'unknown';
const transcriptPath = input.transcript_path || '';

function formatOrchestratorEvent(event) {
  if (!event || typeof event !== 'object') return '';
  const payload = event.payload || {};
  if (event.type === 'stage_status_change') {
    const note = payload.note ? ` (${payload.note})` : '';
    return `${event.type}: ${payload.stage || 'unknown'} -> ${payload.status || 'unknown'}${note}`;
  }
  if (payload.stage && payload.reason) {
    return `${event.type}: ${payload.stage} (${payload.reason})`;
  }
  if (payload.stage) {
    return `${event.type}: ${payload.stage}`;
  }
  return event.type || 'unknown_event';
}

// Create work log directory
const logDir = path.join(cwd, '.claude', 'logs');
fs.mkdirSync(logDir, { recursive: true });

// Generate log filename
const now = new Date();
const dateStr = now.toISOString().split('T')[0].replace(/-/g, '');
const logFile = path.join(logDir, `session-${dateStr}-${sessionId.substring(0, 8)}.md`);

// Get project info
const projectName = path.basename(cwd);

// Build log content
let logContent = '';

logContent += `# 📝 Work Log - ${projectName}\n`;
logContent += `\n`;
logContent += `**Session ID**: ${sessionId}\n`;
logContent += `**Time**: ${common.formatDateTime(now)}\n`;
logContent += `**Directory**: ${cwd}\n`;
logContent += `\n`;

// Git change statistics
logContent += `## 📊 Session Changes\n`;
const gitInfo = common.getGitInfo(cwd);
const changesDetails = gitInfo.is_repo ? common.getChangesDetails(cwd) : { added: 0, modified: 0, deleted: 0 };

if (gitInfo.is_repo) {
  logContent += `**Branch**: ${gitInfo.branch}\n`;
  logContent += `\n`;
  logContent += '```\n';

  if (gitInfo.has_changes) {
    for (const change of gitInfo.changes) {
      logContent += `${change}\n`;
    }
  } else {
    logContent += 'No changes\n';
  }

  logContent += '```\n';

  // Change statistics
  logContent += `\n`;
  logContent += '| Type | Count |\n';
  logContent += '|------|------|\n';
  logContent += `| Added | ${changesDetails.added} |\n`;
  logContent += `| Modified | ${changesDetails.modified} |\n`;
  logContent += `| Deleted | ${changesDetails.deleted} |\n`;
} else {
  logContent += 'Not a Git repository\n';
}

logContent += `\n`;

// Smart suggestions
if (gitInfo.has_changes) {
  logContent += `## 💡 Suggested Actions\n`;
  logContent += `\n`;

  const typeAnalysis = common.analyzeChangesByType(cwd);

  if (changesDetails.modified > 0 || changesDetails.added > 0) {
    logContent += '- Review changes with code review tools\n';
  }
  if (typeAnalysis.test_files > 0) {
    logContent += '- Test files changed, remember to run test suite\n';
  }
  if (typeAnalysis.docs_files > 0) {
    logContent += '- Documentation updated, ensure sync with code\n';
  }
  if (typeAnalysis.sql_files > 0) {
    logContent += '- SQL files changed, ensure all database scripts are updated\n';
  }
  if (typeAnalysis.service_files > 0) {
    logContent += '- New Service/Controller added, remember to update API docs\n';
  }
  if (typeAnalysis.config_files > 0) {
    logContent += '- Config files modified, check if environment variables need updating\n';
  }

  logContent += `\n`;
}

// Read transcript to extract key operations (if available)
if (transcriptPath && fs.existsSync(transcriptPath)) {
  try {
    const transcript = fs.readFileSync(transcriptPath, 'utf8');
    const toolMatches = transcript.match(/Tool used: [A-Z][a-z]*/g) || [];

    if (toolMatches.length > 0) {
      // Count tool usage
      const toolCounts = {};
      for (const match of toolMatches) {
        const tool = match.replace('Tool used: ', '');
        toolCounts[tool] = (toolCounts[tool] || 0) + 1;
      }

      // Sort and take top 10
      const sortedTools = Object.entries(toolCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

      if (sortedTools.length > 0) {
        logContent += `## 🔧 Key Operations\n`;
        logContent += `\n`;

        for (const [tool, count] of sortedTools) {
          logContent += `| ${tool} | ${count} times |\n`;
        }

        logContent += `\n`;
      }
    }
  } catch {
    // Ignore errors
  }
}

// Orchestrator summary
try {
  const orchestrator = require('../scripts/lib/orchestrator');
  const activeRun = orchestrator.loadActiveRun({ cwd });
  if (activeRun) {
    let stages = null;
    try {
      stages = orchestrator.loadStages({ cwd });
    } catch {
      stages = null;
    }

    const currentStageId = activeRun.current_stage;
    const currentStage = (activeRun.stages || {})[currentStageId] || {};
    const currentStageLabel = stages
      ? ((stages.stages.find((stage) => stage.id === currentStageId) || {}).label || currentStageId)
      : currentStageId;
    const staleStages = Object.entries(activeRun.stages || {})
      .filter(([, state]) => state.status === 'stale')
      .map(([stageId]) => stageId);
    const currentArtifacts = ((activeRun.artifacts || {})[currentStageId]) || {};
    const trackedFilesCount = Array.isArray(currentArtifacts.tracked_files)
      ? currentArtifacts.tracked_files.length
      : Object.keys(currentArtifacts.fingerprints || {}).length;

    logContent += `## Orchestrator Status\n`;
    logContent += `\n`;
    logContent += `- Run: ${activeRun.id} - ${activeRun.title}\n`;
    logContent += `- Current stage: ${currentStageLabel} (${currentStageId}) [${currentStage.status || 'unknown'}]\n`;
    if (activeRun.venue || (activeRun.inputs && activeRun.inputs.venue)) {
      logContent += `- Venue: ${activeRun.venue || activeRun.inputs.venue}\n`;
    }
    if (staleStages.length > 0) {
      logContent += `- Stale stages: ${staleStages.join(', ')}\n`;
    }
    if (trackedFilesCount > 0) {
      logContent += `- Tracked files in current stage: ${trackedFilesCount}\n`;
    }
    logContent += `\n`;

    const eventsPath = path.join(
      orchestrator.getOrchestratorRoot({ cwd }),
      'runs',
      activeRun.id,
      'events.ndjson'
    );
    if (fs.existsSync(eventsPath)) {
      const events = fs.readFileSync(eventsPath, 'utf8')
        .split('\n')
        .filter(Boolean)
        .slice(-5)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter(Boolean);

      if (events.length > 0) {
        logContent += `## Recent Orchestrator Events\n`;
        logContent += `\n`;
        for (const event of events) {
          logContent += `- ${event.timestamp}: ${formatOrchestratorEvent(event)}\n`;
        }
        logContent += `\n`;
      }
    }
  }
} catch {
  // Ignore orchestrator errors in session summary
}

// Next steps
logContent += `## 🎯 Next Steps\n`;
logContent += `\n`;

// Git commit suggestions
if (gitInfo.has_changes) {
  logContent += '- ⚠️ Uncommitted changes detected, consider committing first:\n';
  logContent += '  ```bash\n';
  logContent += '  git add . && git commit -m "feat: xxx"\n';
  logContent += '  ```\n';
} else {
  logContent += '- ✅ Working directory clean, ready for new tasks\n';
}

// Todo reminders
const todoInfo = common.getTodoInfo(cwd);
if (todoInfo.found) {
  logContent += `- Update todos: ${todoInfo.file} (${todoInfo.pending} pending)\n`;
}

// CLAUDE.md memory update check
const homeDir = os.homedir();
const claudeMdCheck = common.checkClaudeMdUpdate(homeDir);

if (claudeMdCheck.needsUpdate) {
  logContent += `- ⚠️ **CLAUDE.md memory needs updating** (${claudeMdCheck.changedFiles.length} source files changed)\n`;
  logContent += `  Run "/update-memory" to sync latest memory\n`;

  // Record change details to log
  logContent += `\n`;
  logContent += `### CLAUDE.md Change Details\n`;
  logContent += `\n`;
  logContent += `| Type | File | Modified |\n`;
  logContent += `|------|------|----------|\n`;
  for (const file of claudeMdCheck.changedFiles.slice(0, 10)) {
    logContent += `| ${file.type} | ${file.relativePath} | ${file.mtime} |\n`;
  }
  if (claudeMdCheck.changedFiles.length > 10) {
    logContent += `| ... | ${claudeMdCheck.changedFiles.length - 10} more files | ... |\n`;
  }
} else {
  logContent += `- ✅ CLAUDE.md memory is up to date\n`;
}

logContent += '- View context snapshot: `cat .claude/session-context-*.md`\n';
logContent += `\n`;

// Write log file
fs.writeFileSync(logFile, logContent, 'utf8');

// Build message to display to user
let displayMsg = '\n---\n';
displayMsg += '✅ **Session ended** | Work log saved\n\n';
displayMsg += '**Changes**: ';

if (gitInfo.is_repo) {
  if (gitInfo.has_changes) {
    displayMsg += `${gitInfo.changes_count} files\n\n`;
    displayMsg += '**Suggested actions**:\n';
    displayMsg += `- View log: cat .claude/logs/${path.basename(logFile)}\n`;
    displayMsg += '- Commit code: git add . && git commit -m "feat: xxx"\n';
  } else {
    displayMsg += 'None\n\nWorking directory clean ✅\n';
  }
} else {
  displayMsg += 'Not a Git repository\n';
}

// Add CLAUDE.md update reminder to display message
if (claudeMdCheck.needsUpdate) {
  displayMsg += '\n**⚠️ CLAUDE.md memory needs updating**\n';
  displayMsg += `- ${claudeMdCheck.changedFiles.length} source files changed\n`;
  displayMsg += '- Run `/update-memory` to sync latest memory\n';
}

displayMsg += '\n---';

const result = {
  continue: true,
  systemMessage: displayMsg
};

console.log(JSON.stringify(result));

process.exit(0);
