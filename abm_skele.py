from collections import ChainMap, deque
import random
import math
from bisect import bisect

class Agent():
    def __init__(self, name, maxhist=None):
        """Represents a patient.

        :param name: _description_
        :param maxhist: _description_, defaults to None
        """
        self.id = name
        self.infected = False
        self.location = None
        self.history = deque(maxlen=maxhist)

    def move_to(self, location):
        """Record a movement of the agent to a different location.
        We use history as a lookup, so it contains the current position.
        """
        self.location = location
        self.history.append(self.location)

class Location():
    def __init__(self, name):
        self.id = name
        self.occupants = set()

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

class AgentMover():
    """Mapping of movement probabilities of an agent"""

    def __init__(self, data=None):
        self.move_probs = ChainMap()

    def add_move_probs(self, new_map):
        """Add a new map onto the front of the chain of lookups of move probs"""
        compiled_map = {
            key: CompiledOutcomes(value)
            for key, value in new_map.items()
        }
        self.move_probs = self.move_probs.new_child(compiled_map)

    def next_location(self, agent):
        """Generate a realisation of the next location an agent moves to, 
        given their infection state and their movement history
        """
        agent_state = (agent.infected, *reversed(agent.history))
        movement_outcomes = self.move_probs[agent_state]
        return movement_outcomes.weighted_choice()

class ABM():
    def __init__(self):
        self.agents = dict() # map of agent name to Agent
        self.locations = dict() # map of location name to Location
        self.mover = AgentMover() # map of how people move around
        self.t = 0

    def move(self, agent, location):
        agent_obj = self.agents[agent]
        loc_obj = self.locations[location]

        old_loc = self.locations.get(agent_obj.location, None)
        if old_loc:
            old_loc.occupants.remove(agent)
        agent_obj.move_to(location)
        loc_obj.occupants.add(agent)

    def do_next_move(self, agent):
        agent_obj = self.agents[agent]
        next_loc = self.mover.next_location(agent_obj)
        self.move(agent, next_loc)

    def add_agent(self, agent):
        self.agents[agent.id] = agent

    def add_location(self, location):
        self.locations[location.id] = location

    def add_generic_agents(self, n):
        digits = int(math.ceil(math.log10(n)))
        for i in range(n):
            agent = Agent(f"agent_{i:{digits}.0}")
            self.add_agent(agent)

    def move_all_to(self, location):
        for agent in self.agents:
            self.move(agent, location)