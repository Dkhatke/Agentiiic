"""judge_ideas: a separate model scores the ideator's candidates blind. The model that wrote the ideas tends to rate its
own ideas highly, so the scores come from a different call (ideally a different model, see `[judge]` in the config)."""
from __future__ import annotations

import json
import re

from .. import notes
from .builtin import _S, _clip, _obj
from .registry import Registry, Risk, Tool, ToolContext, ToolError

JUDGE_PROMPT = """You are a strict reviewer of project ideas proposed for one specific person. Score each idea on three
1-5 scales. Be critical: most ideas deserve 2-4, and a 5 must be earned.
Feasibility: 5 = working MVP in a weekend on free tiers; 3 = 2-4 weeks; 1 = needs resources the person lacks.
Usefulness: 5 = they would use it weekly; 1 = a demo nobody needs.
Learning: 5 = teaches a skill they want; 1 = nothing new.
Novelty is not a criterion: unfamiliar is not the same as good.
{context}
Ideas:
{ideas}

Reply with only a JSON array, one object per idea in the same order:
[{{"n": 1, "feasibility": 3, "usefulness": 4, "learning": 5, "reason": "one short sentence"}}]"""

SCALES = ("feasibility", "usefulness", "learning")


def parse_scores(text: str, count: int) -> list[dict] | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        rows = json.loads(m.group(0))
    except ValueError:
        return None
    out = []
    for r in rows if isinstance(rows, list) else []:
        try:
            scores = {k: min(5, max(1, int(r[k]))) for k in SCALES}
            out.append({"n": int(r["n"]), **scores, "reason": str(r.get("reason", ""))[:200]})
        except (KeyError, TypeError, ValueError):
            return None
    return out if out and all(1 <= r["n"] <= count for r in out) else None


def judge_ideas(ctx: ToolContext, ideas: str) -> str:
    lines = [l.strip() for l in ideas.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ToolError("pass one idea per line, formatted 'title: one-sentence pitch' (at least 2 ideas)")
    if ctx.judge is None:
        raise ToolError("no judge model is available; score the ideas yourself with the rubric")
    numbered = "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1))
    context = notes.load(ctx.root).strip()
    prompt = JUDGE_PROMPT.format(ideas=numbered, context=f"\nAbout the person:\n{context}\n" if context else "")
    reply = ctx.judge(prompt)
    scores = parse_scores(reply, len(lines))
    if scores is None:
        return "judge reply could not be parsed; raw reply:\n" + _clip(reply)
    rows = []
    for r in sorted(scores, key=lambda r: -(r["usefulness"] + r["learning"] + r["feasibility"])):
        keep = "KEEP" if r["feasibility"] >= 3 else "DROP (feasibility<3)"
        rows.append(f"{r['n']}. {lines[r['n'] - 1][:80]} | F{r['feasibility']} U{r['usefulness']} L{r['learning']} "
                    f"| {keep} | {r['reason']}")
    return "Independent scores (best first):\n" + "\n".join(rows)


def register(reg: Registry) -> None:
    reg.register(Tool("judge_ideas", "Have a separate model score candidate ideas blind (feasibility, usefulness, learning). "
                      "Pass one idea per line as 'title: pitch'. Use these scores instead of scoring your own ideas.",
                      _obj({"ideas": _S}, ["ideas"]), Risk.READ, judge_ideas))
