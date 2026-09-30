"""RED tests for DSPy 3.x compatibility in the evolution CLI.

The upstream repo (last touched June 2026) targets an older DSPy API. These
tests pin the *current* contract so the fixes cannot silently regress:

1. `dspy.GEPA` no longer accepts `max_steps` -> the optimizer must be built
   with a budget argument that DSPy 3.x actually exposes.
2. DSPy 3.x calls a GEPA metric with FIVE positional args
   (gold, pred, trace, pred_name, pred_trace); the repo's metric takes three.
3. The MIPROv2 fallback path is unreachable dead weight unless optuna exists,
   so a GEPA failure must not silently degrade to a broken fallback.
"""

import inspect

import pytest


def _dspy():
    return pytest.importorskip("dspy")


# ── 1. GEPA budget argument ────────────────────────────────────────────────

def test_gepa_budget_kwarg_is_supported_by_installed_dspy():
    """Whatever kwarg the CLI passes to dspy.GEPA must exist on DSPy 3.x."""
    dspy = _dspy()
    params = inspect.signature(dspy.GEPA.__init__).parameters
    # At least one of these budget knobs must be accepted; `max_steps` is gone.
    assert "max_steps" not in params, (
        "test is stale: dspy.GEPA grew max_steps back"
    )
    assert {"max_metric_calls", "max_full_evals"} & set(params), (
        "dspy.GEPA exposes no metric-call budget knob"
    )


# ── 2. metric arity ─────────────────────────────────────────────────────────

def test_repo_metric_is_not_directly_gepa_compatible():
    """Documents WHY an adapter is needed: the repo metric takes 3 args."""
    from evolution.core.fitness import skill_fitness_metric

    n = len(inspect.signature(skill_fitness_metric).parameters)
    assert n == 3, "repo metric signature changed; re-check the GEPA adapter"


def test_gepa_adapter_accepts_five_positional_args():
    """The adapter the CLI builds must bind 5 positional args, like DSPy 3.x."""
    from evolution.core.fitness import make_gepa_metric, skill_fitness_metric

    adapted = make_gepa_metric(skill_fitness_metric)
    sig = inspect.signature(adapted)
    positional = [
        p for p in sig.parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert len(positional) >= 5, (
        f"adapter binds {len(positional)} positional args, DSPy 3.x needs 5"
    )
    # And it must not blow up when DSPy probes the signature with Nones.
    sig.bind(None, None, None, None, None)  # raises TypeError if arity is wrong


def test_gepa_adapter_delegates_scoring_unchanged():
    """Arity is adapted; the score must still come from the wrapped metric."""
    from evolution.core.fitness import make_gepa_metric

    calls = []

    def three_arg_metric(gold, pred, trace=None):
        calls.append((gold, pred, trace))
        return 0.42

    adapted = make_gepa_metric(three_arg_metric)
    assert adapted("GOLD", "PRED", "TRACE", "pred_name", "pred_trace") == 0.42
    assert calls == [("GOLD", "PRED", "TRACE")]


def test_llm_judge_metric_scores_composite_and_falls_back(monkeypatch):
    """LLM-as-judge must return the composite, and a judge blow-up must fall
    back to the heuristic instead of silently zeroing the rollout."""
    import evolution.core.fitness as fitness_mod
    from evolution.core.fitness import make_llm_judge_metric, skill_fitness_metric
    from evolution.core.config import EvolutionConfig

    config = EvolutionConfig()
    gold = type("Ex", (), {"task_input": "do x", "expected_behavior": "y z w"})()
    pred = type("P", (), {"output": "y z w q"})()

    class FakeScore:
        composite = 0.77

    # Patch BEFORE building the metric: it constructs its judge up front.
    monkeypatch.setattr(
        fitness_mod, "LLMJudge",
        lambda cfg: type("J", (), {"score": lambda self, **kw: FakeScore()})(),
    )
    assert make_llm_judge_metric(config)(gold, pred, None) == 0.77

    # Judge raises -> heuristic fallback, never an exception
    def _boom(self, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fitness_mod, "LLMJudge",
        lambda cfg: type("J", (), {"score": _boom})(),
    )
    expected = skill_fitness_metric(gold, pred, None)
    assert make_llm_judge_metric(config)(gold, pred, None) == expected


def test_llm_judge_metric_receives_skill_text_when_trace_carries_it(monkeypatch):
    """The judge must see the skill text it is scoring against — without it the
    rubric 'did it follow the procedure' has no procedure to compare to."""
    import evolution.core.fitness as fitness_mod
    from evolution.core.fitness import make_llm_judge_metric
    from evolution.core.config import EvolutionConfig

    seen = {}

    class FakeScore:
        composite = 0.5

    def _score(self, **kw):
        seen.update(kw)
        return FakeScore()

    monkeypatch.setattr(
        fitness_mod, "LLMJudge",
        lambda cfg: type("J", (), {"score": _score})(),
    )
    metric = make_llm_judge_metric(EvolutionConfig())
    trace = type("T", (), {"skill_text": "ALWAYS CHECK MEMAVAILABLE"})()
    metric(
        type("Ex", (), {"task_input": "t", "expected_behavior": "e"})(),
        type("P", (), {"output": "o"})(),
        trace,
    )
    assert seen["skill_text"] == "ALWAYS CHECK MEMAVAILABLE"


def test_llm_judge_metric_scores_empty_output_zero():
    """Empty agent output must score 0 without calling the judge."""
    from evolution.core.fitness import make_llm_judge_metric
    from evolution.core.config import EvolutionConfig

    metric = make_llm_judge_metric(EvolutionConfig())
    gold = type("Ex", (), {"task_input": "t", "expected_behavior": "e"})()
    pred = type("P", (), {"output": "   "})()
    assert metric(gold, pred, None) == 0.0


# ── 3. MIPROv2 fallback must not be the silent failure path ────────────────

def test_cli_builds_gepa_with_supported_budget():
    """The optimizer factory must produce a real GEPA without raising, and the
    budget kwarg it passes must be one DSPy 3.x actually accepts.

    A `*args, **kwargs` spy is deliberately NOT used: it would erase the very
    signature the factory introspects, which is what this test must exercise.
    GEPA refuses to construct without a reflection LM, so one is supplied.
    """
    import dspy
    from evolution.skills import evolve_skill

    opt = evolve_skill.build_gepa_optimizer(
        metric=lambda *a: 0.0,
        iterations=10,
        reflection_lm=dspy.LM("openai/dummy-model"),
    )
    assert isinstance(opt, dspy.GEPA)

    # A real GEPA with a real budget must report a non-trivial budget.
    budget = getattr(opt, "max_metric_calls", None)
    if budget is None:
        budget = getattr(opt, "max_full_evals", None)
    assert budget, "GEPA built without any budget knob"
    assert budget > 0


def test_build_gepa_optimizer_requires_reflection_lm():
    """DSPy 3.x GEPA asserts on a missing reflection LM; the factory must not
    swallow that into a silent fallback."""
    from evolution.skills import evolve_skill

    with pytest.raises(AssertionError):
        evolve_skill.build_gepa_optimizer(
            metric=lambda *a: 0.0, iterations=5, reflection_lm=None
        )


def test_build_gepa_optimizer_refuses_unknown_dspy(monkeypatch):
    """If a future DSPy drops every budget knob, fail loudly instead of
    starting an unbounded optimization."""
    import inspect as _inspect
    import dspy
    from evolution.skills import evolve_skill

    bare = _inspect.Signature(
        [_inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    monkeypatch.setattr(dspy.GEPA, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(dspy.GEPA, "__signature__", bare, raising=False)
    with pytest.raises(TypeError, match="no budget knob"):
        evolve_skill.build_gepa_optimizer(lambda *a: 0.0, iterations=5)
