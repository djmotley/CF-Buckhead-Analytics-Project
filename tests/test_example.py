from cf_buckhead_analytics import add_one, random_add_one


def test_add_one() -> None:
    assert add_one(2) == 3


def test_random_add_one() -> None:
    results = {random_add_one(2) for _ in range(100)}
    assert results == {3, 4}  # Expecting both 3 and 4 due to randomness
