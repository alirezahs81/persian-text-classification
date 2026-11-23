from src.preprocess import PersianTextNormalizer, normalize_persian


def test_ye_ke_normalization():
    assert "ی" in normalize_persian("پيگيري")
    assert "ک" in normalize_persian("كلك")


def test_digits_and_diacritics():
    out = normalize_persian("پرداخت ۱۲۳۴ تومانً")
    assert "1234" in out
    assert "ً" not in out


def test_urls_removed():
    out = normalize_persian("ببینید https://example.com/x مشکل دارم")
    assert "http" not in out
    assert "مشکل" in out


def test_sklearn_transformer():
    n = PersianTextNormalizer()
    result = n.fit_transform(["اينترنت قطع است"])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] != ""
