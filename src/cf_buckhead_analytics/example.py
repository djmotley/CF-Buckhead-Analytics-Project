from __future__ import annotations

import random


def add_one(x: int) -> int:
    """Tiny example function used by tests."""
    return x + 1


def random_add_one(x: int) -> int:
    """Adds one to x, but randomly adds two instead to simulate flakiness."""
    if random.random() < 0.1:
        return x + 2
    return x + 1
