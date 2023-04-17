import random
from bisect import bisect
from collections import ChainMap
from typing import Hashable
from .abm import AgentMover

class CompiledOutcomes():
    def __init__(self, data=None):
        self.outcomes = []
        self.cdf = []
        self.total = 0

        if data is not None:
            self.compile(data)
    
    def compile(self, pdict):
        """Construct the CDF for the outcome map"""
        for k,v in pdict.items():
            self.outcomes.append(k)
            self.total += v
            self.cdf.append(self.total)

    def weighted_choice(self):
        """Provided by Raymond Hettinger on stackoverflow
        O(log(n)) lookup on a compiled CDF object
        """
        x = random.random() * self.total
        i = bisect(self.cdf, x)
        return self.outcomes[i]

class MarkovMover(AgentMover):
    """Mapping of movement probabilities of an agent"""

    def __init__(self, *data):
        self.move_probs = ChainMap()
        for datum in data:
            self.add_move_probs(datum)

    def __repr__(self):
        return "AgentMover()"

    def add_move_probs(self, new_map):
        """Add a new map onto the front of the chain of lookups of move probs"""
        compiled_map = {
            key: CompiledOutcomes(value)
            for key, value in new_map.items()
        }
        self.move_probs = self.move_probs.new_child(compiled_map)

    def next_location(self, state: Hashable):
        """Generate a realisation of the next location an agent in the given state moves to
        """
        movement_outcomes = self.move_probs[state]
        return movement_outcomes.weighted_choice()