"""
Neighborhood structures used by VNS / RVNS.

N_k represents the k-th neighborhood in the VNS neighborhood family.
Larger k typically = larger / more disruptive move.

The family used here (matching the GA report's mutation variants):
    N_1: 1-opt  (single city swap)
    N_2: or-opt (single city insertion)
    N_3: 2-opt  (segment reversal)
    N_4: 3-opt  (two-segment reversal, sampled)
    N_5: double-bridge (four-segment cyclic shuffle)
"""

import random
from typing import List


def shake_swap(tour: List[int], rng: random.Random) -> List[int]:
    """N_1: swap two random positions."""
    n = len(tour)
    i, j = rng.sample(range(n), 2)
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def shake_or_opt(tour: List[int], rng: random.Random) -> List[int]:
    """N_2: move one city to another position."""
    n = len(tour)
    i, j = rng.sample(range(n), 2)
    new_tour = tour.copy()
    city = new_tour.pop(i)
    target = j + 1 if j > i else j
    new_tour.insert(target, city)
    return new_tour


def shake_two_opt(tour: List[int], rng: random.Random) -> List[int]:
    """N_3: reverse a segment."""
    n = len(tour)
    i, j = sorted(rng.sample(range(n), 2))
    new_tour = tour.copy()
    new_tour[i:j + 1] = list(reversed(new_tour[i:j + 1]))
    return new_tour


def shake_three_opt(tour: List[int], rng: random.Random) -> List[int]:
    """N_4: pick three break points and apply a random 3-opt reconnection."""
    n = len(tour)
    a, b, c = sorted(rng.sample(range(n), 3))
    new_tour = tour.copy()
    pattern = rng.randint(0, 3)
    if pattern == 0:
        new_tour[a:b + 1] = list(reversed(new_tour[a:b + 1]))
    elif pattern == 1:
        new_tour[b:c + 1] = list(reversed(new_tour[b:c + 1]))
    elif pattern == 2:
        new_tour[a:b + 1] = list(reversed(new_tour[a:b + 1]))
        new_tour[b:c + 1] = list(reversed(new_tour[b:c + 1]))
    return new_tour


def shake_double_bridge(tour: List[int], rng: random.Random) -> List[int]:
    """
    N_5: double-bridge move — split into 4 segments and reconnect in
    a non-sequential way. This is the classic LKH-style kick.
    """
    n = len(tour)
    if n < 8:
        # Fall back to a 3-opt shake for small instances
        return shake_three_opt(tour, rng)

    # Pick three random split points, sort them
    splits = sorted(rng.sample(range(1, n), 3))
    a, b, c = splits
    seg1 = tour[:a]
    seg2 = tour[a:b]
    seg3 = tour[b:c]
    seg4 = tour[c:]
    # Standard double-bridge pattern
    new_tour = seg1 + seg3 + seg2 + seg4
    return new_tour


# Ordered list of neighborhood shake functions
NEIGHBORHOOD_FAMILY = [
    ('N1-swap', shake_swap),
    ('N2-or-opt', shake_or_opt),
    ('N3-2-opt', shake_two_opt),
    ('N4-3-opt', shake_three_opt),
    ('N5-double-bridge', shake_double_bridge),
]
