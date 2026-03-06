/**
 * Orchestrator Runtime Library 测试
 * 使用 Node.js 内置 test runner (node:test)
 */

'use strict';

const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');

const orchestrator = require('../scripts/lib/orchestrator');

// 获取项目根目录（stages.json 所在位置）
const PROJECT_ROOT = path.resolve(__dirname, '..');

/**
 * 创建临时目录模拟项目环境
 * 复制 orchestrator/stages.json 到临时目录
 */
function createTempProject() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'orch-test-'));
  // 复制 stages.json
  const srcStages = path.join(PROJECT_ROOT, 'orchestrator', 'stages.json');
  const dstDir = path.join(tmpDir, 'orchestrator');
  fs.mkdirSync(dstDir, { recursive: true });
  fs.copyFileSync(srcStages, path.join(dstDir, 'stages.json'));
  return tmpDir;
}

/**
 * 递归删除目录
 */
function cleanupDir(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    // 忽略
  }
}

describe('getOrchestratorRoot', () => {
  it('returns .claude/orchestrator under cwd', () => {
    const root = orchestrator.getOrchestratorRoot({ cwd: '/tmp/test-project' });
    assert.equal(root, '/tmp/test-project/.claude/orchestrator');
  });
});

describe('loadStages', () => {
  it('loads stages.json from orchestrator directory', () => {
    const stages = orchestrator.loadStages({ cwd: PROJECT_ROOT });
    assert.ok(stages.stages);
    assert.ok(Array.isArray(stages.stages));
    assert.ok(stages.stages.length >= 10);

    const ids = stages.stages.map((s) => s.id);
    assert.ok(ids.includes('intake'));
    assert.ok(ids.includes('literature'));
    assert.ok(ids.includes('experiments'));
    assert.ok(ids.includes('self_review'));
  });
});

describe('initRun', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('creates active-run.json and run.json', () => {
    const run = orchestrator.initRun({
      cwd: tmpDir,
      title: 'Test Research',
      profile: 'test-profile',
      venue: 'NeurIPS',
    });

    // 验证 run 对象
    assert.ok(run.id);
    assert.equal(run.title, 'Test Research');
    assert.equal(run.profile, 'test-profile');
    assert.equal(run.venue, 'NeurIPS');
    assert.equal(run.current_stage, 'intake');
    assert.ok(run.stages);
    assert.equal(run.stages.intake.status, 'pending');
    // venue/profile 同时存在于 top-level 和 inputs
    assert.equal(run.inputs.venue, 'NeurIPS');
    assert.equal(run.inputs.profile, 'test-profile');

    // 验证文件存在
    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const activeRun = JSON.parse(fs.readFileSync(path.join(root, 'active-run.json'), 'utf8'));
    assert.equal(activeRun.run_id, run.id);

    const runFile = JSON.parse(
      fs.readFileSync(path.join(root, 'runs', run.id, 'run.json'), 'utf8')
    );
    assert.equal(runFile.id, run.id);
    assert.equal(runFile.title, 'Test Research');

    // 验证 events.ndjson 存在
    const eventsPath = path.join(root, 'runs', run.id, 'events.ndjson');
    assert.ok(fs.existsSync(eventsPath));
  });

  it('throws if title is missing', () => {
    assert.throws(() => orchestrator.initRun({ cwd: tmpDir }), /title is required/);
  });

  it('generates unique run IDs', () => {
    const run1 = orchestrator.initRun({ cwd: tmpDir, title: 'Run 1' });
    const run2 = orchestrator.initRun({ cwd: tmpDir, title: 'Run 2' });
    assert.notEqual(run1.id, run2.id);
  });
});

describe('loadActiveRun', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('returns null when no active run exists', () => {
    const run = orchestrator.loadActiveRun({ cwd: tmpDir });
    assert.equal(run, null);
  });

  it('returns run object when active run exists', () => {
    const created = orchestrator.initRun({ cwd: tmpDir, title: 'Active Run Test' });
    const loaded = orchestrator.loadActiveRun({ cwd: tmpDir });
    assert.ok(loaded);
    assert.equal(loaded.id, created.id);
    assert.equal(loaded.title, 'Active Run Test');
  });
});

describe('markStage', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
    orchestrator.initRun({ cwd: tmpDir, title: 'Mark Stage Test' });
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('updates stage status and timestamps', () => {
    const run = orchestrator.markStage({
      cwd: tmpDir,
      stageId: 'intake',
      status: 'in_progress',
    });
    assert.equal(run.stages.intake.status, 'in_progress');
    assert.ok(run.stages.intake.started_at);
    assert.equal(run.current_stage, 'intake');
  });

  it('marks stage as done with completed_at', () => {
    orchestrator.markStage({ cwd: tmpDir, stageId: 'intake', status: 'in_progress' });
    const run = orchestrator.markStage({
      cwd: tmpDir,
      stageId: 'intake',
      status: 'done',
    });
    assert.equal(run.stages.intake.status, 'done');
    assert.ok(run.stages.intake.completed_at);
  });

  it('rejects invalid status', () => {
    assert.throws(
      () => orchestrator.markStage({ cwd: tmpDir, stageId: 'intake', status: 'invalid' }),
      /Invalid status/
    );
  });

  it('records note when provided', () => {
    const run = orchestrator.markStage({
      cwd: tmpDir,
      stageId: 'intake',
      status: 'blocked',
      note: 'Waiting for user input',
    });
    assert.equal(run.stages.intake.note, 'Waiting for user input');
  });

  it('rejects unknown stageId (typo protection)', () => {
    assert.throws(
      () => orchestrator.markStage({ cwd: tmpDir, stageId: 'analysys', status: 'in_progress' }),
      /Unknown stageId/
    );
  });

  it('appends event to events.ndjson', () => {
    orchestrator.markStage({ cwd: tmpDir, stageId: 'intake', status: 'in_progress' });

    const active = orchestrator.loadActiveRun({ cwd: tmpDir });
    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const eventsPath = path.join(root, 'runs', active.id, 'events.ndjson');
    const lines = fs.readFileSync(eventsPath, 'utf8').trim().split('\n').filter(Boolean);
    assert.ok(lines.length >= 1);

    const event = JSON.parse(lines[lines.length - 1]);
    assert.equal(event.type, 'stage_status_change');
    assert.equal(event.payload.stage, 'intake');
    assert.equal(event.payload.status, 'in_progress');
  });
});

describe('setStageStatus (rollback)', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
    orchestrator.initRun({ cwd: tmpDir, title: 'Rollback Test' });
    // 设置几个阶段为 done
    orchestrator.markStage({ cwd: tmpDir, stageId: 'intake', status: 'done' });
    orchestrator.markStage({ cwd: tmpDir, stageId: 'literature', status: 'done' });
    orchestrator.markStage({ cwd: tmpDir, stageId: 'proposal', status: 'done' });
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('clears downstream done stages on rollback to pending', () => {
    const run = orchestrator.setStageStatus({
      cwd: tmpDir,
      stageId: 'literature',
      status: 'pending',
      reason: 'User requested rollback',
    });
    assert.equal(run.stages.literature.status, 'pending');
    // proposal（下游）应该被清除
    assert.equal(run.stages.proposal.status, 'pending');
    // intake（上游）应该不变
    assert.equal(run.stages.intake.status, 'done');
  });

  it('marks stage as stale and clears downstream', () => {
    const run = orchestrator.setStageStatus({
      cwd: tmpDir,
      stageId: 'intake',
      status: 'stale',
      reason: 'Artifact changed',
    });
    assert.equal(run.stages.intake.status, 'stale');
    assert.equal(run.stages.literature.status, 'pending');
    assert.equal(run.stages.proposal.status, 'pending');
  });

  it('syncs current_stage to rolled-back stage', () => {
    // current_stage 应在 proposal（最后一个 markStage in_progress/done 的阶段）
    orchestrator.markStage({ cwd: tmpDir, stageId: 'proposal', status: 'in_progress' });
    const before = orchestrator.loadActiveRun({ cwd: tmpDir });
    assert.equal(before.current_stage, 'proposal');

    // 回滚到 literature
    orchestrator.setStageStatus({
      cwd: tmpDir,
      stageId: 'literature',
      status: 'pending',
      reason: 'Rollback test',
    });
    const after = orchestrator.loadActiveRun({ cwd: tmpDir });
    assert.equal(after.current_stage, 'literature');
  });
});

describe('fingerprintFiles', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
    // 创建测试文件
    fs.writeFileSync(path.join(tmpDir, 'test-file.md'), '# Test content\n', 'utf8');
    fs.writeFileSync(path.join(tmpDir, 'another.txt'), 'Hello world\n', 'utf8');
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('returns SHA256 hashes for existing files', () => {
    const result = orchestrator.fingerprintFiles({
      cwd: tmpDir,
      paths: ['test-file.md', 'another.txt'],
    });
    assert.ok(result['test-file.md']);
    assert.ok(result['test-file.md'].startsWith('sha256:'));
    assert.ok(result['another.txt']);
    assert.ok(result['another.txt'].startsWith('sha256:'));
  });

  it('skips missing files', () => {
    const result = orchestrator.fingerprintFiles({
      cwd: tmpDir,
      paths: ['test-file.md', 'nonexistent.md'],
    });
    assert.ok(result['test-file.md']);
    assert.equal(result['nonexistent.md'], undefined);
  });

  it('produces consistent hashes', () => {
    const r1 = orchestrator.fingerprintFiles({ cwd: tmpDir, paths: ['test-file.md'] });
    const r2 = orchestrator.fingerprintFiles({ cwd: tmpDir, paths: ['test-file.md'] });
    assert.equal(r1['test-file.md'], r2['test-file.md']);
  });

  it('detects content changes', () => {
    const r1 = orchestrator.fingerprintFiles({ cwd: tmpDir, paths: ['test-file.md'] });
    fs.writeFileSync(path.join(tmpDir, 'test-file.md'), '# Changed content\n', 'utf8');
    const r2 = orchestrator.fingerprintFiles({ cwd: tmpDir, paths: ['test-file.md'] });
    assert.notEqual(r1['test-file.md'], r2['test-file.md']);
  });
});

describe('collectTrackedFiles', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('includes contract-declared file artifacts for non-writeup stages', () => {
    const run = orchestrator.initRun({ cwd: tmpDir, title: 'Tracked Files Test' });
    fs.writeFileSync(path.join(tmpDir, 'literature-review.md'), '# Literature\n', 'utf8');
    fs.writeFileSync(path.join(tmpDir, 'references.bib'), '@article{a,title={A}}\n', 'utf8');

    const tracked = orchestrator.collectTrackedFiles({
      cwd: tmpDir,
      run,
      stageId: 'literature',
    });

    assert.deepEqual(tracked, ['literature-review.md', 'references.bib']);
  });

  it('expands main tex dependencies recursively for writeup', () => {
    const paperDir = path.join(tmpDir, 'paper');
    const sectionsDir = path.join(paperDir, 'sections');
    const figuresDir = path.join(paperDir, 'figures');
    const refsDir = path.join(paperDir, 'refs');

    fs.mkdirSync(sectionsDir, { recursive: true });
    fs.mkdirSync(figuresDir, { recursive: true });
    fs.mkdirSync(refsDir, { recursive: true });

    fs.writeFileSync(
      path.join(paperDir, 'main.tex'),
      [
        '\\documentclass{article}',
        '\\input{sections/intro}',
        '\\include{sections/method}',
        '\\bibliography{refs/main}',
        '\\addbibresource{refs/extra.bib}',
        '\\begin{document}',
        '\\includegraphics[width=0.9\\\\linewidth]{figures/overview}',
        '\\end{document}',
        '',
      ].join('\n'),
      'utf8'
    );
    fs.writeFileSync(path.join(sectionsDir, 'intro.tex'), 'Intro\n\\input{details}\n', 'utf8');
    fs.writeFileSync(path.join(sectionsDir, 'details.tex'), 'Details\n', 'utf8');
    fs.writeFileSync(path.join(sectionsDir, 'method.tex'), 'Method\n', 'utf8');
    fs.writeFileSync(path.join(refsDir, 'main.bib'), '@article{main,title={Main}}\n', 'utf8');
    fs.writeFileSync(path.join(refsDir, 'extra.bib'), '@article{extra,title={Extra}}\n', 'utf8');
    fs.writeFileSync(path.join(figuresDir, 'overview.pdf'), '%PDF-1.4\n', 'utf8');

    let run = orchestrator.initRun({ cwd: tmpDir, title: 'Writeup Tracking Test' });
    run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: {
        artifacts: {
          writeup: {
            main_tex: 'paper/main.tex',
            figures_dir: 'paper/figures',
          },
        },
      },
    });

    const tracked = orchestrator.collectTrackedFiles({
      cwd: tmpDir,
      run,
      stageId: 'writeup',
    });

    assert.deepEqual(tracked, [
      'paper/figures/overview.pdf',
      'paper/main.tex',
      'paper/refs/extra.bib',
      'paper/refs/main.bib',
      'paper/sections/details.tex',
      'paper/sections/intro.tex',
      'paper/sections/method.tex',
    ]);
  });
});

describe('fingerprintStageArtifacts', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('returns tracked_files and fingerprints for a stage', () => {
    fs.mkdirSync(path.join(tmpDir, 'paper', 'sections'), { recursive: true });
    fs.writeFileSync(path.join(tmpDir, 'paper', 'main.tex'), '\\input{sections/intro}\n', 'utf8');
    fs.writeFileSync(path.join(tmpDir, 'paper', 'sections', 'intro.tex'), 'Intro\n', 'utf8');

    let run = orchestrator.initRun({ cwd: tmpDir, title: 'Stage Fingerprint Test' });
    run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: {
        artifacts: {
          writeup: {
            main_tex: 'paper/main.tex',
            figures_dir: 'paper/figures',
          },
        },
      },
    });

    const result = orchestrator.fingerprintStageArtifacts({
      cwd: tmpDir,
      run,
      stageId: 'writeup',
    });

    assert.deepEqual(result.tracked_files, ['paper/main.tex', 'paper/sections/intro.tex']);
    assert.equal(Object.keys(result.fingerprints).length, 2);
    assert.ok(result.fingerprints['paper/main.tex'].startsWith('sha256:'));
    assert.ok(result.fingerprints['paper/sections/intro.tex'].startsWith('sha256:'));
  });

  it('changes fingerprints when a referenced writeup dependency changes', () => {
    fs.mkdirSync(path.join(tmpDir, 'paper', 'sections'), { recursive: true });
    fs.writeFileSync(path.join(tmpDir, 'paper', 'main.tex'), '\\input{sections/intro}\n', 'utf8');
    fs.writeFileSync(path.join(tmpDir, 'paper', 'sections', 'intro.tex'), 'Intro\n', 'utf8');

    let run = orchestrator.initRun({ cwd: tmpDir, title: 'Fingerprint Change Test' });
    run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: {
        artifacts: {
          writeup: {
            main_tex: 'paper/main.tex',
            figures_dir: 'paper/figures',
          },
        },
      },
    });

    const before = orchestrator.fingerprintStageArtifacts({
      cwd: tmpDir,
      run,
      stageId: 'writeup',
    });

    fs.writeFileSync(path.join(tmpDir, 'paper', 'sections', 'intro.tex'), 'Intro changed\n', 'utf8');

    const after = orchestrator.fingerprintStageArtifacts({
      cwd: tmpDir,
      run,
      stageId: 'writeup',
    });

    assert.notEqual(
      before.fingerprints['paper/sections/intro.tex'],
      after.fingerprints['paper/sections/intro.tex']
    );
  });
});

describe('appendEvent', () => {
  let tmpDir;
  let runId;

  beforeEach(() => {
    tmpDir = createTempProject();
    const run = orchestrator.initRun({ cwd: tmpDir, title: 'Event Test' });
    runId = run.id;
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('appends valid NDJSON entries', () => {
    orchestrator.appendEvent({
      cwd: tmpDir,
      runId,
      type: 'test_event',
      payload: { key: 'value' },
    });
    orchestrator.appendEvent({
      cwd: tmpDir,
      runId,
      type: 'another_event',
      payload: { count: 42 },
    });

    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const eventsPath = path.join(root, 'runs', runId, 'events.ndjson');
    const lines = fs.readFileSync(eventsPath, 'utf8').trim().split('\n').filter(Boolean);
    assert.ok(lines.length >= 2);

    const event1 = JSON.parse(lines[lines.length - 2]);
    assert.equal(event1.type, 'test_event');
    assert.equal(event1.payload.key, 'value');
    assert.ok(event1.timestamp);

    const event2 = JSON.parse(lines[lines.length - 1]);
    assert.equal(event2.type, 'another_event');
    assert.equal(event2.payload.count, 42);
  });
});

describe('withRunLock', () => {
  let tmpDir;
  let runId;

  beforeEach(() => {
    tmpDir = createTempProject();
    const run = orchestrator.initRun({ cwd: tmpDir, title: 'Lock Test' });
    runId = run.id;
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('executes function under lock and releases it', () => {
    let executed = false;
    const result = orchestrator.withRunLock({ cwd: tmpDir, runId }, () => {
      executed = true;
      return 'success';
    });
    assert.equal(executed, true);
    assert.equal(result, 'success');

    // 锁应该已释放
    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const lockPath = path.join(root, 'runs', runId, '.lock');
    assert.equal(fs.existsSync(lockPath), false);
  });

  it('detects concurrent lock and fails fast', () => {
    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const lockPath = path.join(root, 'runs', runId, '.lock');

    // 手动创建锁
    fs.writeFileSync(lockPath, JSON.stringify({ pid: 99999, timestamp: Date.now() }));

    assert.throws(
      () => orchestrator.withRunLock({ cwd: tmpDir, runId }, () => {}),
      /locked by another process/
    );
  });

  it('releases lock even if function throws', () => {
    assert.throws(() =>
      orchestrator.withRunLock({ cwd: tmpDir, runId }, () => {
        throw new Error('test error');
      })
    );

    // 锁应该已释放
    const root = orchestrator.getOrchestratorRoot({ cwd: tmpDir });
    const lockPath = path.join(root, 'runs', runId, '.lock');
    assert.equal(fs.existsSync(lockPath), false);
  });
});

describe('updateRun', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = createTempProject();
    orchestrator.initRun({ cwd: tmpDir, title: 'Update Test' });
  });

  afterEach(() => {
    cleanupDir(tmpDir);
  });

  it('merges patch into run state', () => {
    const run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: {
        inputs: { venue: 'ICML', topic: 'RLHF' },
      },
    });
    assert.equal(run.inputs.venue, 'ICML');
    assert.equal(run.inputs.topic, 'RLHF');
    assert.ok(run.updated_at);
  });

  it('deep merges nested objects', () => {
    orchestrator.updateRun({
      cwd: tmpDir,
      patch: { artifacts: { literature: { draft: true } } },
    });
    const run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: { artifacts: { literature: { reviewed: true } } },
    });
    assert.equal(run.artifacts.literature.draft, true);
    assert.equal(run.artifacts.literature.reviewed, true);
  });

  it('rejects non-object patch', () => {
    assert.throws(() => orchestrator.updateRun({ cwd: tmpDir, patch: 'bad' }), /plain object/);
    assert.throws(() => orchestrator.updateRun({ cwd: tmpDir, patch: null }), /plain object/);
    assert.throws(() => orchestrator.updateRun({ cwd: tmpDir, patch: [1] }), /plain object/);
  });

  it('ignores prototype pollution keys', () => {
    const run = orchestrator.updateRun({
      cwd: tmpDir,
      patch: { __proto__: { polluted: true }, inputs: { safe: true } },
    });
    assert.equal(run.inputs.safe, true);
    assert.equal(({}).polluted, undefined);
  });
});
