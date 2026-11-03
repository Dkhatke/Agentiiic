"""Project mode with a scripted planner and worker: planning, checks, checkpoints, the stuck prompt and resuming."""
import json
import subprocess

import pytest

from agentic import project as pj
from agentic.tools import ToolContext, ToolError

PLAN = json.dumps([{"title": "core", "goal": "make a.txt", "check": "test -f a.txt"},
                   {"title": "extra", "goal": "make b.txt", "check": "test -f b.txt"}])


def ctx(root):
    return ToolContext(root, 10, False, False)


def git_repo(root):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)


def commits(root):
    r = subprocess.run(["git", "log", "--format=%s"], cwd=root, capture_output=True, text=True)
    return r.stdout.split("\n")[:-1] if r.returncode == 0 else []


@pytest.mark.parametrize("reply", ["Here you go:\n" + PLAN + "\nGood luck", PLAN])
def test_parse_plan_finds_the_array_in_prose(reply):
    plan = pj.parse_plan(reply)
    assert [m.title for m in plan] == ["core", "extra"] and plan[0].check == "test -f a.txt"


@pytest.mark.parametrize("reply", ["", "no json", "[1, 2]", '[{"title": "x"}]', "[{bad", '{"title": "a", "goal": "b", "check": "c"}'])
def test_parse_plan_rejects_junk(reply):
    assert pj.parse_plan(reply) is None


def test_make_plan_retries_once_then_fails():
    replies = iter(["sorry, here is prose", PLAN])
    conventions, plan = pj.make_plan("spec", lambda p: next(replies))
    assert len(plan) == 2 and conventions == ""
    with pytest.raises(ToolError, match="planner"):
        pj.make_plan("spec", lambda p: "still prose")


def test_plan_is_capped_and_saved_and_loaded(tmp_path):
    many = json.dumps([{"title": f"m{i}", "goal": "g", "check": "true"} for i in range(20)])
    plan = pj.parse_plan(many)
    assert len(plan) == pj.MAX_MILESTONES
    pj.save(tmp_path, "the spec", plan, "use cli.py")
    spec, conventions, back = pj.load(tmp_path)
    assert spec == "the spec" and back == plan and conventions == "use cli.py"
    assert "project.json" in (tmp_path / ".agentic" / ".gitignore").read_text()
    assert pj.load(tmp_path / "nowhere") is None


def test_happy_path_checks_and_checkpoints_each_milestone(tmp_path):
    git_repo(tmp_path)
    c = ctx(tmp_path)
    assert "agentic/project" in pj.prepare_git(c, "agentic/project")
    tasks = []

    def worker(task):
        tasks.append(task)
        (tmp_path / ("a.txt" if "make a.txt" in task else "b.txt")).write_text("x")
        return "done"
    plan = pj.parse_plan(PLAN)
    out = pj.run_project("spec", plan, c, worker, lambda m, o: pytest.fail("should not get stuck"), on_event=lambda e: None)
    assert "2/2" in out and [m.status for m in plan] == ["done", "done"]
    assert commits(tmp_path) == ["milestone 2: extra", "milestone 1: core"]
    assert "first milestone" in tasks[0] and "Already finished" in tasks[1] and "- core" in tasks[1]  # fresh session, told what exists
    assert "project.json" not in subprocess.run(["git", "ls-files"], cwd=tmp_path, capture_output=True, text=True).stdout


def test_failing_milestone_retries_then_asks_and_uses_the_hint(tmp_path):
    tasks, asked = [], []

    def worker(task):
        tasks.append(task)
        if "Hint from the person: create a.txt" in task:
            (tmp_path / "a.txt").write_text("x")
        return "tried"
    plan = pj.parse_plan(PLAN)[:1]
    out = pj.run_project("spec", plan, ctx(tmp_path), worker, lambda m, o: asked.append(o) or "create a.txt", on_event=lambda e: None)
    assert len(asked) == 1 and len(tasks) == pj.ATTEMPTS + 1
    assert "A previous attempt failed" in tasks[1] and "exit=1" in tasks[1]  # the retry sees the failing check
    assert plan[0].status == "done" and "1/1" in out


def test_skip_and_quit_then_continue(tmp_path):
    plan = pj.parse_plan(PLAN)
    out = pj.run_project("spec", plan, ctx(tmp_path), lambda t: "", lambda m, o: "s", on_event=lambda e: None)
    assert [m.status for m in plan] == ["skipped", "skipped"] and "0/2" in out

    plan = pj.parse_plan(PLAN)
    out = pj.run_project("spec", plan, ctx(tmp_path), lambda t: "", lambda m, o: "q", on_event=lambda e: None)
    assert "--continue" in out and plan[0].status == "pending"
    spec, _, saved = pj.load(tmp_path)
    assert saved[0].attempts == pj.ATTEMPTS

    def worker(task):  # resuming builds only what is still pending
        (tmp_path / ("a.txt" if "make a.txt" in task else "b.txt")).write_text("x")
    out = pj.run_project(spec, saved, ctx(tmp_path), worker, lambda m, o: "q", on_event=lambda e: None)
    assert "2/2" in out


def test_undo_stashes_the_failed_attempt_and_retries_fresh(tmp_path):
    git_repo(tmp_path)
    c = ctx(tmp_path)
    pj.prepare_git(c, "agentic/project")
    (tmp_path / "keep.txt").write_text("k")
    pj.checkpoint(c, "start")
    answers = iter(["u", "s"])

    def worker(task):
        (tmp_path / "junk.txt").write_text("bad attempt")
    plan = pj.parse_plan(PLAN)[:1]
    pj.run_project("spec", plan, c, worker, lambda m, o: next(answers), on_event=lambda e: None)
    stash = subprocess.run(["git", "stash", "list"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "undone milestone attempt" in stash and (tmp_path / "keep.txt").exists()


def test_prepare_git_never_lands_on_main(tmp_path):
    git_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x")
    c = ctx(tmp_path)
    assert "untouched" in pj.prepare_git(c, "agentic/project")
    fresh = tmp_path / "new"
    fresh.mkdir()
    assert "started a git repository" in pj.prepare_git(ctx(fresh), "agentic/project")


def test_check_command_respects_protected_files(tmp_path):
    (tmp_path / ".env").write_text("K=v")
    ok, out = pj.run_check(ctx(tmp_path), "cat .env")
    assert not ok and "protected" in out


def test_plan_object_with_conventions_and_single_milestone():
    reply = 'Plan: {"conventions": "CLI is `python3 app.py [--db PATH] <command>`", "milestones": [' \
            '{"title": "core", "goal": "g", "check": "true"}]}'
    conventions, plan = pj.make_plan("spec", lambda p: reply)
    assert "--db PATH" in conventions and [m.title for m in plan] == ["core"]
    one = '[{"title": "solo", "goal": "g", "check": "true"}]'  # a bare one-item array also looks like an object inside
    assert [m.title for m in pj.parse_plan(one)] == ["solo"]


def test_later_milestones_get_conventions_and_earlier_summaries(tmp_path):
    tasks = []

    def worker(task):
        tasks.append(task)
        (tmp_path / ("a.txt" if "make a.txt" in task else "b.txt")).write_text("x")
        return "Added app.py with add_book(conn, title); CLI: python3 app.py [--db PATH] add TITLE"
    plan = pj.parse_plan(PLAN)
    pj.run_project("spec", plan, ctx(tmp_path), worker, lambda m, o: "q", on_event=lambda e: None, conventions="DB in library.db")
    assert all("DB in library.db" in t for t in tasks)
    assert "add_book(conn, title)" in tasks[1] and "add_book" not in tasks[0]
    assert plan[0].notes.startswith("Added app.py") and pj.load(tmp_path)[1] == "DB in library.db"
