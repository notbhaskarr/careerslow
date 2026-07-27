"""Tests for resume draft edit application."""

from src.utils.resume_draft import AcceptedEdit, apply_edits


SAMPLE = """Experience
Amdocs
Functional Test Engineer
Oversaw and managed the analysis, design, implementation, and execution of test cases and test suites
Applied domain expertise to maintain testing efficiency across 15+ major releases
Testing & Quality: Selenium, Defect Management, Unix, SQL
"""


def test_edit_exact_line_replace():
    result = apply_edits(
        SAMPLE,
        [
            AcceptedEdit(
                action="edit",
                target_line="Oversaw and managed the analysis, design, implementation, and execution of test cases and test suites",
                text="Managed end-to-end defect lifecycle and test execution across 15+ releases",
            )
        ],
    )
    assert "Managed end-to-end defect lifecycle" in result
    assert "Oversaw and managed the analysis" not in result


def test_edit_fuzzy_match():
    result = apply_edits(
        SAMPLE,
        [
            AcceptedEdit(
                action="edit",
                target_line="Applied domain expertise to maintain testing efficiency",
                text="Led risk-based test planning across 15+ major telecom releases",
            )
        ],
    )
    assert "Led risk-based test planning" in result


def test_add_appends_bullets():
    result = apply_edits(
        SAMPLE,
        [AcceptedEdit(action="add", text="Conducted API contract testing with Postman")],
    )
    assert result.strip().endswith("Conducted API contract testing with Postman")
    assert "• Conducted API contract testing" in result


def test_skips_empty_add():
    result = apply_edits(SAMPLE, [AcceptedEdit(action="add", text="   ")])
    assert result == SAMPLE
