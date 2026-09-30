"""The skill text must live in the channel GEPA actually mutates.

GEPA seeds its candidate space from
`{name: pred.signature.instructions for ...}` (dspy/teleprompt/gepa/gepa.py) and
rewrites those strings reflectively. The original `SkillModule` kept the skill
in a plain `self.skill_text` attribute next to the predictor, so GEPA optimised
the signature docstring and the "evolved" skill came back byte-identical to the
baseline — a silent no-op that still reports success.

These tests pin the property that makes Phase 1 actually evolve a skill.
"""

import dspy
import pytest

from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)


def _instructions_of(module: SkillModule) -> str:
    # ChainOfThought wraps a Predict; the signature GEPA mutates is the inner one.
    return module.predictor.predict.signature.instructions


# ── the skill text is inside the instructions ───────────────────────────────

def test_skill_text_is_carried_inside_predictor_instructions():
    m = SkillModule("DO THE THING CAREFULLY")
    instr = _instructions_of(m)
    assert "DO THE THING CAREFULLY" in instr, (
        "skill text must live in signature.instructions — that is the only "
        "text GEPA mutates"
    )


def test_gepa_seed_candidate_would_contain_the_skill():
    """Reproduce GEPA's own candidate construction: whatever it seeds is what
    it can evolve. The skill must appear there."""
    m = SkillModule("UNIQUE_SKILL_TOKEN_42")
    # This mirrors gepa.py: seed_candidate = {name: pred.signature.instructions}
    seed_candidate = {n: p.signature.instructions for n, p in m.named_predictors()}
    assert any("UNIQUE_SKILL_TOKEN_42" in v for v in seed_candidate.values()), (
        "GEPA's seed candidate does not contain the skill text; GEPA would "
        "evolve the docstring instead of the skill"
    )


# ── round-trip extraction ───────────────────────────────────────────────────

def test_extract_skill_text_roundtrips_unchanged_text():
    original = "# Title\n\nStep one. Step two."
    m = SkillModule(original)
    assert SkillModule.extract_skill_text(_instructions_of(m)) == original.strip()


def test_extract_skill_text_returns_what_gepa_rewrote():
    """Simulate GEPA replacing the instructions with an evolved variant."""
    m = SkillModule("OLD SKILL BODY")
    m.predictor.predict.signature.instructions = (
        "Complete a task following the provided skill instructions.\n\n"
        f"{SkillModule.SKILL_MARKER}\n"
        "NEW EVOLVED SKILL BODY with more procedure"
    )
    assert SkillModule.extract_skill_text(_instructions_of(m)) == "NEW EVOLVED SKILL BODY with more procedure"


def test_extract_skill_text_tolerates_marker_loss():
    """If reflective mutation rewrote the whole prompt and dropped the marker,
    fall back to the full text instead of returning an empty skill."""
    m = SkillModule("x")
    got = SkillModule.extract_skill_text("A completely rewritten instruction")
    assert got == "A completely rewritten instruction"
    assert got != ""


def test_extract_skill_text_handles_empty_instructions():
    assert SkillModule.extract_skill_text("") == ""
    assert SkillModule.extract_skill_text(None) == ""


# ── the stale-attribute trap the CLI hit ────────────────────────────────────

def test_cli_reads_evolved_text_through_predictor_predict_signature():
    """Regression: the CLI once read `optimized_module.predictor.signature`,
    which does not exist — ChainOfThought wraps a Predict, so the signature
    lives on `.predict`. The whole run died at the extraction step AFTER a
    14-minute optimization, so pin the access path used by the CLI."""
    from evolution.skills.skill_module import SkillModule as SM

    class Optimized:
        def __init__(self, instructions):
            self.skill_text = "BASELINE STALE"
            self.predictor = dspy.ChainOfThought(SM.TaskWithSkill)
            self.predictor.predict.signature.instructions = instructions

    optimized = Optimized(
        f"prefix\n\n{SM.SKILL_MARKER}\nEVOLVED SKILL BODY"
    )
    # The exact expression the CLI must use:
    got = SM.extract_skill_text(
        optimized.predictor.predict.signature.instructions
    )
    assert got == "EVOLVED SKILL BODY"
    # ...and it must NOT be the stale attribute
    assert got != optimized.skill_text


def test_evolved_skill_text_property_reflects_instructions_not_stale_attr():
    """`skill_text` stays at the baseline; the property must read the live
    instructions, which is what the CLI now uses."""
    m = SkillModule("BASELINE")
    m.predictor.predict.signature.instructions = (
        f"prefix\n\n{SkillModule.SKILL_MARKER}\nEVOLVED"
    )
    assert m.skill_text == "BASELINE"          # stale by design
    assert m.evolved_skill_text == "EVOLVED"   # what callers must use


# ── the module still behaves as a DSPy module ───────────────────────────────

def test_forward_passes_skill_text_as_input_field(monkeypatch):
    """The skill must reach the LLM as an input field too, not only via the
    instructions, so the agent actually reads it."""
    m = SkillModule("SKILL BODY HERE")
    seen = {}

    class FakeResult:
        output = "done"

    # ChainOfThought -> Predict.__call__ is the real boundary; patch it there.
    monkeypatch.setattr(
        type(m.predictor.predict), "__call__",
        lambda self, **kwargs: (seen.update(kwargs), FakeResult())[1],
    )
    pred = m(task_input="do it")
    assert seen["skill_instructions"] == "SKILL BODY HERE"
    assert seen["task_input"] == "do it"
    assert pred.output == "done"


# ── size accounting still works on the real shape ──────────────────────────

def test_skill_body_size_matches_extracted_text():
    """The constraint validator budgets by len(skill['body']); extraction must
    return a body of comparable size, not the prompt scaffolding."""
    body = "x" * 5000
    m = SkillModule(body)
    extracted = SkillModule.extract_skill_text(_instructions_of(m))
    assert len(extracted) == 5000


# ── load/find/reassemble unchanged ──────────────────────────────────────────

def test_reassemble_preserves_frontmatter():
    out = reassemble_skill("name: demo\ndescription: d", "BODY")
    assert out.startswith("---")
    assert "name: demo" in out
    assert "BODY" in out
