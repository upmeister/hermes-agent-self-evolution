"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text is the parameter GEPA optimizes, and it is carried in the
    predictor's *instructions* — that is the only channel GEPA mutates. GEPA
    seeds its candidate space from
    `{name: pred.signature.instructions for ...}` and rewrites those strings
    reflectively. A `self.skill_text` attribute sitting next to the predictor
    is invisible to it, so an "optimized" module can come back byte-identical.

    The docstring prefix is load-bearing: `SkillModule.extract_skill_text()`
    splits an evolved instruction back into the skill body, so whatever GEPA
    writes after the marker is the evolved skill.

    Note the access path: `dspy.ChainOfThought` is a `Predict` wrapper, so the
    signature lives on `self.predictor.predict.signature`, and that is also the
    key GEPA reports in `named_predictors()` ("predict").
    """

    SKILL_MARKER = "<!-- SKILL TEXT -->"

    class TaskWithSkill(dspy.Signature):
        """Complete a task following the provided skill instructions.

        You are an AI agent following specific skill instructions to complete a task.
        Read the skill instructions carefully and follow the procedure described.
        """
        skill_instructions: str = dspy.InputField(desc="The skill instructions to follow")
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = skill_text
        self.predictor = dspy.ChainOfThought(self.TaskWithSkill)
        # Put the skill INTO the instructions — the only text GEPA can evolve.
        self._sync_instructions()

    @property
    def _signature(self):
        """The signature GEPA mutates (ChainOfThought -> Predict wrapper)."""
        return self.predictor.predict.signature

    def _sync_instructions(self):
        base = self.TaskWithSkill.instructions
        prefix = base.split(self.SKILL_MARKER)[0].rstrip()
        self._signature.instructions = (
            f"{prefix}\n\n{self.SKILL_MARKER}\n{self.skill_text}"
        )

    @classmethod
    def extract_skill_text(cls, instructions: Optional[str]) -> str:
        """Recover the skill body from an evolved instruction string.

        An evolved instruction may have rewritten the whole prompt, so fall
        back to the full text when the marker is gone rather than returning "".
        """
        if not instructions:
            return ""
        if cls.SKILL_MARKER in instructions:
            return instructions.split(cls.SKILL_MARKER, 1)[1].strip()
        return instructions.strip()

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(
            skill_instructions=self.skill_text,
            task_input=task_input,
        )
        return dspy.Prediction(output=result.output)

    @property
    def evolved_skill_text(self) -> str:
        """The skill text as it currently lives inside the instructions."""
        return self.extract_skill_text(self._signature.instructions)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
