__all__ = ['Agent', 'MemoryAgent', 'Location', 'AgentMover', 'EventType', 'Event', 'ModelState', 'Model']

import dataclasses
import enum
import math
import heapq
from collections import deque, defaultdict
from functools import total_ordering
from typing import Hashable
from abc import ABC, abstractmethod

class HeapList():
    """Class that is used to maintain a sorted list"""
    def __init__(self, data=None):
        if data is None:
            self.heap = []
        else:
            self.heap = list(data)
            heapq.heapify(self.heap)

    def add(self, item):
        """Push item onto heap"""
        heapq.heappush(self.heap, item)

    def __getitem__(self, index):
        return self.heap[index]

    def pop(self):
        """Pop the minimum item (first item)"""
        return heapq.heappop(self.heap)

    def __len__(self):
        return len(self.heap)

class Agent():
    def __init__(self, name: Hashable):
        """Represents a patient or other vector.

        :param name: unique identifier
        """
        self.id = name
        self.infected = False
        self.location = None

    def __repr__(self):
        return f"{self.__class__}(name={self.id})" 

    def __str__(self):
        return f"Agent ({self.id}) [{self.location}] {{{self.infected}}}"

    def move_to(self, location: Hashable):
        """Record a movement of the agent to a different location.
        """
        self.location = location

    @property
    def state(self):
        return self.infected

class MemoryAgent(Agent):

    def __init__(self, name: Hashable, maxhist: int|None=None):
        """Represents a patient ot other vector. Retains a history of previously visited locations.
        
        :param name: unique identifier
        :param maxhist: (max) length of recorded location history, defaults to None (no limit)
        """
        super().__init__(name)
        self.history = deque(maxlen=maxhist)

    def __repr__(self):
        return f"{self.__class__}(name={self.id}, maxhist={self.history.maxlen})"

    def move_to(self, location: Hashable):
        """Record a movement to another location.
        Also inserts the new location into history for lookups
        """
        self.location = location
        self.history.append(location)

    @property
    def state(self):
        return self.infected, *reversed(self.history)

class Location():
    def __init__(self, name: Hashable):
        self.id = name
        self.occupants = set()

    def __repr__(self):
        return f"Location('{self.id}')"

    def __str__(self) -> str:
        return F"Location ({self.id}) [{len(self.occupants)}]"

class AgentMover(ABC):
    """Helper object that can be used to determine the next location an agent moves to
    
    See: markov_mover.MarkovMover for an implementation of a Markov model
    """

    def next_location(self, *args, **kwargs):
        """Returns the next location that an agent would move to"""
        pass

# EventType = enum.Enum('EventType', [])
@total_ordering
class EventType(enum.Enum):
    def __lt__(self, other):
        return self.value < other.value

@dataclasses.dataclass(frozen=True, eq=True, order=True)
class Event():
    """Data object that represents an event that will happen"""
    t: float
    event_type: EventType
    agent: Hashable

class ModelState():
    """State representation of the model.
    Contains the agents, locations, time, queue of events.

    Attributes:
    -----------
    agents : map of agent name to Agent object
    locations: map of location name to Location object
    t: current simulation time
    event_queue: queue (sorted list/deque) of events
    event_map: map of agent name to list of associated queued events
    """
    
    def __init__(self):
        self.agents = dict() # map of agent name to Agent
        self.locations = dict() # map of location name to Location
        self.t = 0
        self.event_queue = HeapList() # list of events
        self.event_map = defaultdict(set) # map of agent to events

        self._event_base = Event

    def __repr__(self) -> str:
        return "ABM()"

    def __str__(self) -> str:
        return f"ABM: [{self.t}] {len(self.agents)} agents, {len(self.locations)} locations, {len(self.event_queue)} events queued."

    def move(self, agent: Hashable, location: Hashable):
        agent_obj = self.agents[agent]
        loc_obj = self.locations[location]

        old_loc = self.locations.get(agent_obj.location, None)
        if old_loc:
            old_loc.occupants.discard(agent)
        agent_obj.move_to(location)
        loc_obj.occupants.add(agent)

    def add_agent(self, agent: Agent):
        self.agents[agent.id] = agent

    def add_location(self, location: Location):
        self.locations[location.id] = location

    def add_generic_agents(self, n: int):
        """Helper utility for adding in random agents"""
        digits = int(math.ceil(math.log10(n)))
        for i in range(n):
            agent = Agent(f"agent_{i:0{digits}}")
            self.add_agent(agent)

    def move_all_to(self, location: Hashable):
        for agent in self.agents:
            self.move(agent, location)

    def add_event(self, t: float, event_type: EventType, agent: Hashable, *args, **kwargs):
        event = self._event_base(t, event_type, agent, *args, **kwargs)
        self.event_queue.add(event)
        self.event_map[agent].add(event)

class ModelHistory(list):
    pass

class Model(ABC):

    def __init__(self):
        self.state = ModelState()
        self.history = ModelHistory()

    @abstractmethod
    def simulate(self, until=None):
        pass