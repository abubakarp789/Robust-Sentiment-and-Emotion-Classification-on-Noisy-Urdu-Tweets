from __future__ import annotations

from src.preprocessing import preprocess_text


def test_preprocessing_removes_emoji_url_and_mention() -> None:
    cleaned = preprocess_text("@user #خوشی https://example.com 😊")

    assert "@user" not in cleaned
    assert "http" not in cleaned
    assert "😊" not in cleaned
    assert "خوشی" in cleaned


def test_hashtag_text_is_preserved() -> None:
    cleaned = preprocess_text("#پاکستان زندہ باد")

    assert "#" not in cleaned
    assert "پاکستان" in cleaned
