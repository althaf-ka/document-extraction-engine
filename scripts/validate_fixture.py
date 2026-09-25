"""Known test2.pdf regression checks, not general document validation."""

import argparse
import json
import re
from pathlib import Path

FIXTURE_SHA256 = "fa7e2312819bcf0d536cd3d2008d21d1a740e62c7452cea126743b65b41a37c0"


def validate(markdown: str, max_watermarks: int = 0) -> dict:
    headings = list(re.finditer(r"(?m)^\s*(?:#+\s*)?Q\s*(\d+)\s*[.:]", markdown))
    questions = {}
    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        )
        number = int(heading.group(1))
        questions[number] = questions.get(number, "") + markdown[heading.end() : end]
    checks = {}
    for number in (1, 8, 15):
        labels = re.findall(r"(?<!\w)\(([1-4])\)(?=\s|$)", questions.get(number, ""))
        checks[f"Q{number} options"] = {
            "passed": sorted(labels) == ["1", "2", "3", "4"],
            "observed": labels,
            "expected": ["1", "2", "3", "4"],
        }
    patterns = {
        "Q2 bounds": (2, [r"\b1600\b", r"\b1800\b"]),
        "Q3 formula": (3, [r"\\frac", r"\\sqrt"]),
        "Q4 pi/2": (
            4,
            [r"(?:\\[dt]?frac\s*\{\s*\\pi\s*\}\s*\{\s*2\s*\}|π\s*/\s*2|\\pi\s*/\s*2)"],
        ),
        "Q6 digit restriction": (
            6,
            [r"digits?\s*\$?\s*1\s*,\s*2\s*(?:,\s*(?:and\s*)?|and\s*)3", r"\bonly\b"],
        ),
        "Q11 matrix": (11, [r"\\begin\{(?:bmatrix|pmatrix|matrix|array)\}"]),
        "Q14 determinant": (14, [r"\\begin\{vmatrix\}"]),
    }
    for name, (number, required) in patterns.items():
        checks[name] = {
            "passed": all(
                re.search(pattern, questions.get(number, ""), re.IGNORECASE) is not None
                for pattern in required
            )
        }
    count = len(re.findall("MathonGo", markdown, re.IGNORECASE))
    checks["MathonGo occurrences"] = {
        "passed": count <= max_watermarks,
        "observed": count,
        "maximum": max_watermarks,
    }
    return {
        "fixture": "test2.pdf",
        "fixture_sha256": FIXTURE_SHA256,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, nargs="+")
    parser.add_argument("--max-watermarks", type=int, default=0)
    args = parser.parse_args()
    reports = {
        str(path): validate(path.read_text(encoding="utf-8"), args.max_watermarks)
        for path in args.markdown
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(report["passed"] for report in reports.values()) else 1)


if __name__ == "__main__":
    main()
