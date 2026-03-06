/**
 * paper-figure-generator metadata tests
 */

'use strict';

const { describe, it, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const PYTHON_ENV = { ...process.env, PYTHONDONTWRITEBYTECODE: '1' };

function cleanupDir(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

function run(cmd, args, opts = {}) {
  const result = spawnSync(cmd, args, {
    encoding: 'utf8',
    ...opts,
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result;
}

describe('paper-figure-generator run metadata', () => {
  const createdDirs = [];

  afterEach(() => {
    while (createdDirs.length > 0) {
      cleanupDir(createdDirs.pop());
    }
  });

  it('derives project_root from the workspace repo instead of an external reference image path', () => {
    const repoDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-gen-repo-'));
    const externalDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-gen-external-'));
    createdDirs.push(repoDir, externalDir);

    run('git', ['init'], { cwd: repoDir });
    run('git', ['config', 'user.name', 'Test User'], { cwd: repoDir });
    run('git', ['config', 'user.email', 'test@example.com'], { cwd: repoDir });

    fs.writeFileSync(path.join(repoDir, 'README.md'), '# test\n', 'utf8');
    run('git', ['add', 'README.md'], { cwd: repoDir });
    run('git', ['commit', '-m', 'init'], { cwd: repoDir });

    const methodDir = path.join(repoDir, 'paper', 'figures', 'system-overview');
    fs.mkdirSync(methodDir, { recursive: true });
    const methodFile = path.join(methodDir, 'method.txt');
    fs.writeFileSync(methodFile, 'test method\n', 'utf8');

    const externalReference = path.join(externalDir, 'sample3.png');
    fs.writeFileSync(externalReference, 'png-placeholder\n', 'utf8');

    const helper = path.join(
      PROJECT_ROOT,
      'skills',
      'paper-figure-generator',
      'scripts',
      'write_run_metadata.py'
    );

    const result = spawnSync(
      'python3',
      [
        helper,
        '--output-dir',
        methodDir,
        '--method-file',
        methodFile,
        '--provider',
        'openrouter',
        '--sam-backend',
        'roboflow',
        '--api-key-source',
        'env',
        '--api-key-var',
        'OPENROUTER_API_KEY',
        '--',
        '--reference_image_path',
        externalReference,
      ],
      {
        cwd: externalDir,
        encoding: 'utf8',
        env: PYTHON_ENV,
      }
    );

    assert.equal(result.status, 0, result.stderr || result.stdout);

    const runJson = JSON.parse(fs.readFileSync(path.join(methodDir, 'run.json'), 'utf8'));
    assert.equal(fs.realpathSync(runJson.project_root), fs.realpathSync(repoDir));
    assert.match(runJson.git_rev, /^[0-9a-f]+$/);
    assert.deepEqual(runJson.args.slice(-2), ['--reference_image_path', externalReference]);
  });

  it('resolves workspace root from repo-local method/output paths even when cwd is outside the repo', () => {
    const repoDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-root-repo-'));
    const externalDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-root-external-'));
    createdDirs.push(repoDir, externalDir);

    run('git', ['init'], { cwd: repoDir });
    run('git', ['config', 'user.name', 'Test User'], { cwd: repoDir });
    run('git', ['config', 'user.email', 'test@example.com'], { cwd: repoDir });
    fs.writeFileSync(path.join(repoDir, 'README.md'), '# test\n', 'utf8');
    run('git', ['add', 'README.md'], { cwd: repoDir });
    run('git', ['commit', '-m', 'init'], { cwd: repoDir });

    const methodDir = path.join(repoDir, 'paper', 'figures', 'overview');
    fs.mkdirSync(methodDir, { recursive: true });
    const methodFile = path.join(methodDir, 'method.txt');
    fs.writeFileSync(methodFile, 'test method\n', 'utf8');

    const helper = path.join(
      PROJECT_ROOT,
      'skills',
      'paper-figure-generator',
      'scripts',
      'workspace_root.py'
    );

    const result = spawnSync(
      'python3',
      [
        helper,
        '--output-dir',
        methodDir,
        '--method-file',
        methodFile,
        '--workspace-cwd',
        externalDir,
      ],
      {
        cwd: externalDir,
        encoding: 'utf8',
        env: PYTHON_ENV,
      }
    );

    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.equal(fs.realpathSync(result.stdout.trim()), fs.realpathSync(repoDir));
  });

  it('loads provider secrets from the workspace root .env when generate.sh is launched outside the repo', () => {
    const repoDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-env-repo-'));
    const externalDir = fs.mkdtempSync(path.join(os.tmpdir(), 'paper-figure-env-external-'));
    createdDirs.push(repoDir, externalDir);

    run('git', ['init'], { cwd: repoDir });
    run('git', ['config', 'user.name', 'Test User'], { cwd: repoDir });
    run('git', ['config', 'user.email', 'test@example.com'], { cwd: repoDir });
    fs.writeFileSync(path.join(repoDir, 'README.md'), '# test\n', 'utf8');
    run('git', ['add', 'README.md'], { cwd: repoDir });
    run('git', ['commit', '-m', 'init'], { cwd: repoDir });

    fs.writeFileSync(path.join(repoDir, '.env'), 'OPENROUTER_API_KEY=workspace-secret\n', 'utf8');

    const methodDir = path.join(repoDir, 'paper', 'figures', 'overview');
    fs.mkdirSync(methodDir, { recursive: true });
    const methodFile = path.join(methodDir, 'method.txt');
    fs.writeFileSync(methodFile, 'test method\n', 'utf8');
    const externalReference = path.join(externalDir, 'sample3.png');
    fs.writeFileSync(externalReference, 'png-placeholder\n', 'utf8');

    const envCaptureFile = path.join(externalDir, 'captured-env.txt');
    const wrapperPath = path.join(externalDir, 'fake-python.sh');
    fs.writeFileSync(
      wrapperPath,
      `#!/usr/bin/env bash
set -euo pipefail
script_path="$1"
shift
case "$(basename "$script_path")" in
  write_run_metadata.py)
    printf '%s' "\${OPENROUTER_API_KEY:-}" > "$AUTOFIGURE_TEST_ENV_FILE"
    exec python3 "$script_path" "$@"
    ;;
  autofigure2.py)
    exit 0
    ;;
  lint_no_title.py)
    exit 0
    ;;
  *)
    echo "unexpected script: $script_path" >&2
    exit 1
    ;;
esac
`,
      { encoding: 'utf8', mode: 0o755 }
    );

    const generateScript = path.join(
      PROJECT_ROOT,
      'skills',
      'paper-figure-generator',
      'scripts',
      'generate.sh'
    );

    const env = { ...process.env };
    delete env.OPENROUTER_API_KEY;
    delete env.BIANXIE_API_KEY;
    delete env.ROBOFLOW_API_KEY;
    delete env.FAL_KEY;
    env.PYTHONDONTWRITEBYTECODE = '1';
    env.AUTOFIGURE_PYTHON = wrapperPath;
    env.AUTOFIGURE_TEST_ENV_FILE = envCaptureFile;

    const result = spawnSync(
      'bash',
      [
        generateScript,
        '--method_file',
        methodFile,
        '--output_dir',
        methodDir,
        '--provider',
        'openrouter',
        '--sam_backend',
        'local',
        '--reference_image_path',
        externalReference,
      ],
      {
        cwd: externalDir,
        env,
        encoding: 'utf8',
        timeout: 10000,
      }
    );

    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.doesNotMatch(
      `${result.stdout}\n${result.stderr}`,
      /fatal: not a git repository/
    );
    assert.equal(fs.readFileSync(envCaptureFile, 'utf8'), 'workspace-secret');

    const runJson = JSON.parse(fs.readFileSync(path.join(methodDir, 'run.json'), 'utf8'));
    assert.equal(fs.realpathSync(runJson.project_root), fs.realpathSync(repoDir));
    assert.deepEqual(runJson.args.slice(-2), ['--reference_image_path', externalReference]);
  });
});
