/**
 * Session summary hook 测试
 */

'use strict';

const { describe, it, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

const orchestrator = require('../scripts/lib/orchestrator');

const PROJECT_ROOT = path.resolve(__dirname, '..');

function createTempProject() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'session-summary-test-'));
  const srcStages = path.join(PROJECT_ROOT, 'orchestrator', 'stages.json');
  const dstDir = path.join(tmpDir, 'orchestrator');
  fs.mkdirSync(dstDir, { recursive: true });
  fs.copyFileSync(srcStages, path.join(dstDir, 'stages.json'));
  return tmpDir;
}

function cleanupDir(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

describe('hooks/session-summary.js', () => {
  const createdDirs = [];

  afterEach(() => {
    while (createdDirs.length > 0) {
      cleanupDir(createdDirs.pop());
    }
  });

  it('writes orchestrator status and recent events into the session log', () => {
    const tmpDir = createTempProject();
    createdDirs.push(tmpDir);

    let run = orchestrator.initRun({ cwd: tmpDir, title: 'Summary Hook Test', venue: 'ICLR' });
    run = orchestrator.markStage({ cwd: tmpDir, stageId: 'intake', status: 'done' });
    run = orchestrator.markStage({ cwd: tmpDir, stageId: 'literature', status: 'in_progress' });
    run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: {
        artifacts: {
          literature: {
            tracked_files: ['literature-review.md', 'references.bib'],
          },
        },
      },
    });

    orchestrator.appendEvent({
      cwd: tmpDir,
      runId: run.id,
      type: 'stage_status_change',
      payload: { stage: 'literature', status: 'in_progress', note: null },
    });
    orchestrator.appendEvent({
      cwd: tmpDir,
      runId: run.id,
      type: 'stale_detected',
      payload: { stage: 'proposal', reason: 'Artifact changed: research-proposal.md' },
    });

    const input = JSON.stringify({
      cwd: tmpDir,
      session_id: 'session-summary-test',
    });

    const result = spawnSync('node', ['hooks/session-summary.js'], {
      cwd: PROJECT_ROOT,
      input,
      encoding: 'utf8',
    });

    assert.equal(result.status, 0, result.stderr);

    const logDir = path.join(tmpDir, '.claude', 'logs');
    const files = fs.readdirSync(logDir).filter((file) => file.endsWith('.md'));
    assert.equal(files.length, 1);

    const logContent = fs.readFileSync(path.join(logDir, files[0]), 'utf8');
    assert.match(logContent, /## Orchestrator Status/);
    assert.match(logContent, /Summary Hook Test/);
    assert.match(logContent, /literature/);
    assert.match(logContent, /## Recent Orchestrator Events/);
    assert.match(logContent, /stage_status_change/);
    assert.match(logContent, /stale_detected/);
  });
});
