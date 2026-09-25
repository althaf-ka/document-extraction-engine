import runpy
from pathlib import Path

validate = runpy.run_path(
    str(Path(__file__).parents[2] / "scripts" / "validate_fixture.py")
)["validate"]


def test_fixture_qa_detects_missing_and_duplicate_choices():
    report = validate(
        "Q1. Question\n(1) A\n(2) B\n(3) C\n(4) D\nQ8. Question\n(1) A\n(2) B\n(3) C\nQ15. Question\n(1) A\n(2) B\n(3) C\n(3) D\n"
    )
    assert report["checks"]["Q1 options"]["passed"]
    assert not report["checks"]["Q8 options"]["passed"]
    assert not report["checks"]["Q15 options"]["passed"]


def test_absent_questions_fail_and_watermarks_are_counted():
    report = validate("MathonGo MathonGo")
    assert not report["passed"]
    assert not report["checks"]["Q14 determinant"]["passed"]
    assert report["checks"]["MathonGo occurrences"]["observed"] == 2
