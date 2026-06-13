from __future__ import annotations

from src.validate_professor_requirements import validate_requirements


def test_every_professor_requirement_has_evidence() -> None:
    result = validate_requirements()

    assert result["ready"], result
    assert result["rubric_components_mapped"] == 6
    assert result["rubric_marks_mapped"] == 50
