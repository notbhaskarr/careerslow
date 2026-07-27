from src.utils.text import extract_candidate_name


def test_extract_candidate_name_from_header():
    raw = "Bhaskar Bhardwaj\nAI Engineer\nbbhardwaj.work@gmail.com"
    assert extract_candidate_name(raw) == "Bhaskar"
