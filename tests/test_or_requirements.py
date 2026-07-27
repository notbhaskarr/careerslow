from src.utils.or_requirements import (
    parse_or_options,
    count_or_matches,
    apply_or_match_score,
)


def test_parse_or_options_playwright_group():
    req = "At least one of: Playwright, Selenium, Cypress, or similar tools"
    assert parse_or_options(req) == ["Playwright", "Selenium", "Cypress"]


def test_parse_or_options_non_or():
    assert parse_or_options("Python 3+") == []


def test_count_or_matches_selenium_only():
    req = "At least one of: Playwright, Selenium, Cypress, or similar tools"
    options = parse_or_options(req)
    evidence = ["Built E2E tests with Selenium WebDriver for checkout flows"]
    assert count_or_matches(options, evidence) == ["Selenium"]


def test_count_or_matches_two_tools():
    req = "At least one of: Playwright, Selenium, Cypress, or similar tools"
    options = parse_or_options(req)
    evidence = [
        "Automated UI flows with Playwright",
        "Legacy suite maintained in Selenium",
    ]
    assert count_or_matches(options, evidence) == ["Playwright", "Selenium"]


def test_count_or_matches_java_not_javascript():
    req = "At least one of: JavaScript, TypeScript, Python, Java, or C#"
    options = parse_or_options(req)
    evidence = ["Shipped React features in TypeScript"]
    assert count_or_matches(options, evidence) == ["TypeScript"]
    assert "Java" not in count_or_matches(options, evidence)


def test_apply_or_match_score_one_match():
    assert apply_or_match_score(1, 10) == 9
    assert apply_or_match_score(1, 5) == 9


def test_apply_or_match_score_two_matches():
    assert apply_or_match_score(2, 7) == 10
    assert apply_or_match_score(3, 4) == 10


def test_apply_or_match_score_zero_caps_inflated():
    assert apply_or_match_score(0, 10) == 7
    assert apply_or_match_score(0, 4) == 4
