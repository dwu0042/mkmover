from time import perf_counter
from collections import Counter
import pytest
import abm_skele as ab
from typing import Sequence, Callable
from functools import wraps
import random

@pytest.fixture
def move_probs():
    return {
        ('S', 'A'): {'B': 0.7, 'C': 0.3}, 
        ('S', 'B'): {'A': 0.4, 'C': 0.6}, 
        ('S', 'C'): {'A': 0.8, 'B': 0.2},
        ('S', 'B', 'A'): {'A': 0.2, 'C': 0.8},
        ('S', 'C', 'A'): {'A': 0.4, 'B': 0.6},
        ('S', 'A', 'B'): {'B': 0.1, 'C': 0.9},
        ('S', 'C', 'B'): {'A': 0.9, 'B': 0.1},
        ('S', 'A', 'C'): {'B': 0.7, 'C': 0.3},
        ('S', 'B', 'C'): {'A': 0.8, 'C': 0.2},
    }

@pytest.fixture
def mover(move_probs):
    # construct trasition probability
    mover = ab.AgentMover()
    mover.add_move_probs(move_probs)
    return mover

@pytest.fixture
def harold():
    # make the agent
    agent = ab.Agent('Harold', maxhist=2)
    agent.infected = 'S'
    return agent

@pytest.fixture
def locations():
    # make the locations
    location_a = ab.Location('A')
    location_b = ab.Location('B')
    location_c = ab.Location('C')
    return (location_a, location_b, location_c)

@pytest.fixture
def abm(mover : ab.AgentMover, harold : ab.Agent, locations : Sequence[ab.Location]):
    # make the simulation control object
    abm = ab.ABM()
    abm.add_agent(harold)
    for location in locations:
        abm.add_location(location)
    abm.mover = mover
    # move harold to A
    abm.move(harold.id, locations[0].id)
    return abm

def timeit(func : Callable):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        t0 = perf_counter()
        result = func(*args, **kwargs)
        t1 = perf_counter()
        print(f'{func.__name__} took {t1-t0} s')
        return result

    return wrapped_func

@timeit
def test_move_next_hist_1(abm, harold, move_probs):
    """Tests that Mover lookup with 1-history works"""

    # force clear hist
    harold.history = []
    abm.move(harold.id, 'A')

    new_locations = [abm.mover.next_location(harold) for _ in range(100_000)]
    counts = Counter(new_locations)
    expected = move_probs['S', 'A']['B'] / move_probs['S', 'A']['C']

    assert counts['B'] / counts['C'] == pytest.approx(expected, rel=1e-2)

@timeit
def test_move_next_hist_2(abm, harold, move_probs):
    """Tests that Mover lookup with 2-history, and abm manual moving works"""

    # this force moves the agent into an AB history (agent has 2-hist max)
    for loc in ['C', 'C', 'A', 'B']:
        abm.move(harold.id, loc)

    new_locations = [abm.mover.next_location(harold) for _ in range(100_000)]
    counts = Counter(new_locations)
    expected = move_probs['S', 'B', 'A']['C'] / move_probs['S', 'B', 'A']['A']

    assert counts['C'] / counts['A'] == pytest.approx(expected, rel=1.5e-2)

@timeit
def test_next_move(abm, harold, move_probs):
    """Tests ABM control over mover + movement update works"""

    # hand derived stationary distribution
    expected = {'A': 0.3557047, 'B': 0.29614094, 'C': 0.34815436}
    record = {'A': 0, 'B': 0, 'C': 0}
    N = 20_000
    for _ in range(N):
        # cycle length must be co-prime with 2 and 3 to prevent no-return states
        # since we do not allow for agents to dwell in one state (must move)
        for _ in range(random.choice([5, 7, 11])):
            abm.do_next_move(harold.id)
        record[harold.location] += 1
    assert [r/N for r in record.values()] == pytest.approx(list(expected.values()), rel=0.02)