
from agents.agent import Agent
from typing import Callable


class RandomAgent(Agent):
    name = 'random_agent'

    def __init__(self, sample_fn: Callable):
        self.sample_fn = sample_fn

    def sample_actions(self, *args, **kwargs):
        return self.sample_fn(*args, **kwargs)

    def update_state(self, state: dict) -> None:
        pass
