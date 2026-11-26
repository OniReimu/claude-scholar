"""Tests for prereg_check.py. Run: uv run --with pytest pytest test_prereg_check.py"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "prereg_check.py"
TEMPLATE = HERE.parent / "assets" / "EXP-YYYY-NNN"

FILLED_RULES = """## Decision rules

### Claim 1: the new loss improves accuracy

- **Primary metric and direction:** accuracy, higher is better.
- **Comparator:** tuned cross-entropy baseline
- **Supported:** the paired-bootstrap 95% CI excludes zero on the favourable side, and at least 3 seeds agree.
- **Refuted:** the CI lies entirely on the unfavourable side.
- **Refuted (equivalent):** the CI lies entirely inside ±0.5 points.
- **Inconclusive:** every other outcome, including p < 0.05 on a different test.
- **Resampling unit:** test example
- **Seeds:** 3, entering through initialisation.
- **After an inconclusive verdict:** none
- **Worth pursuing:** 1 point

"""


def run(record):
    r = subprocess.run([sys.executable, str(SCRIPT), str(record)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def make_record(root, frozen=True, filled=True):
    record = root / "experiments" / "EXP-2026-001"
    shutil.copytree(TEMPLATE, record)
    plan = record / "analysis-plan.md"
    text = plan.read_text(encoding="utf-8")
    if frozen:
        text = text.replace("**Status:** draft", "**Status:** Frozen", 1)
    if filled:
        start = text.index("## Decision rules")
        end = text.index("## Multiplicity")
        text = text[:start] + FILLED_RULES + text[end:]
    plan.write_text(text, encoding="utf-8")
    return record


def make_repo(root):
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")


def commit_all(root):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "plan")


def test_missing_file(tmp_path):
    record = make_record(tmp_path)
    (record / "protocol.md").unlink()
    code, out = run(record)
    assert code == 1 and "REFUSED: protocol.md not found" in out


def test_draft_status(tmp_path):
    record = make_record(tmp_path, frozen=False)
    code, out = run(record)
    assert code == 1 and "not frozen" in out


def test_unfilled_rules(tmp_path):
    record = make_record(tmp_path, filled=False)
    code, out = run(record)
    assert code == 1 and "unfilled" in out


def test_shipped_template_is_refused(tmp_path):
    record = tmp_path / "EXP-YYYY-NNN"
    shutil.copytree(TEMPLATE, record)
    code, out = run(record)
    assert code == 1 and "not frozen" in out and "unfilled" in out


def test_uncommitted(tmp_path):
    make_repo(tmp_path)
    record = make_record(tmp_path)
    code, out = run(record)
    assert code == 1 and "not tracked" in out


def test_modified_after_commit(tmp_path):
    make_repo(tmp_path)
    record = make_record(tmp_path)
    commit_all(tmp_path)
    with open(record / "analysis-plan.md", "a", encoding="utf-8") as f:
        f.write("\nA late change.\n")
    code, out = run(record)
    assert code == 1 and "differs from HEAD" in out


def test_pass(tmp_path):
    make_repo(tmp_path)
    record = make_record(tmp_path)
    commit_all(tmp_path)
    code, out = run(record)
    assert code == 0 and out.startswith("OK EXP-2026-001") and "committed at " in out


def test_pass_outside_git(tmp_path):
    record = make_record(tmp_path)
    code, out = run(record)
    assert code == 0 and "not in a git work tree" in out


def test_angle_brackets_in_prose_pass(tmp_path):
    record = make_record(tmp_path)
    plan = record / "analysis-plan.md"
    plan.write_text(plan.read_text(encoding="utf-8").replace(
        "- **Worth pursuing:** 1 point",
        "- **Worth pursuing:** <1 point when cost >0, see <https://example.org/method>"),
        encoding="utf-8")
    code, out = run(record)
    assert code == 0, out


def test_second_claim_without_verdict_lines_is_refused(tmp_path):
    record = make_record(tmp_path)
    plan = record / "analysis-plan.md"
    plan.write_text(plan.read_text(encoding="utf-8").replace(
        "## Multiplicity", "### Claim 2: it is also faster\n\n- **Comparator:** baseline\n\n## Multiplicity"),
        encoding="utf-8")
    code, out = run(record)
    assert code == 1 and "unfilled" in out
