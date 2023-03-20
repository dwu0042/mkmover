from collections import ChainMap, deque
import random
import math
from bisect import bisect
import dataclasses
import enum
import sortedcontainers

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

    def __repr__(self):
        return f"Agent('{self.id}', maxhist={self.history.maxlen})"

    def __str__(self):
        return f"Agent ({self.id}) [{self.location}] {{{self.infected}}}"

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

    def __repr__(self):
        return f"Location('{self.id}')"

    def __str__(self) -> str:
        return F"Location ({self.id}) [{len(self.occupants)}]"

    def _repr_pretty_(self, p, cycle):
       p.text(str(self) if not cycle else '...')


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
        if data is not None:
            self.add_move_probs(data)

    def __repr__(self):
        return "AgentMover()"

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

class EventType(enum.Enum):
    Move = enum.auto()
    Infect = enum.auto()

@dataclasses.dataclass(order=True)
class Event():
    t: float
    event_type: EventType
    agent: str

class ABM():
    def __init__(self):
        self.agents = dict() # map of agent name to Agent
        self.locations = dict() # map of location name to Location
        self.mover = AgentMover() # map of how people move around
        self.t = 0
        self.queue = sortedcontainers.SortedList()

    def __repr__(self) -> str:
        return "ABM()"

    def __str__(self) -> str:
        return f"ABM: [{self.t}] {len(self.agents)} agents, {len(self.locations)} locations, {len(self.queue)} events queued."

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
            agent = Agent(f"agent_{i:0{digits}}")
            self.add_agent(agent)

    def move_all_to(self, location):
        for agent in self.agents:
            self.move(agent, location)

    def add_event(self, t, event_type, agent):
        self.queue.add(Event(t, event_type, agent))

    def do_potential_infect(self, agent):
        loc = self.agents[agent].location
        targets = self.locations[loc].occupants
        if len(targets) < 2:
            return
        cands = random.sample(targets, k=2)
        if cands[0] == agent:
            cands.remove(agent)
        cand_obj = self.agents[cands[0]]
        if cand_obj.infected == 'S':
            cand_obj.infected = 'I'

    def handle_next_event(self):
        next_event = self.queue.pop(0)
        self.t = next_event.t
        if next_event.event_type is EventType.Move:
            self.do_next_move(next_event.agent)
        elif next_event.event_type is EventType.Infect:
            self.do_potential_infect(next_event.agent)