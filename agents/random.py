
import numpy as np
from agents.agent import Agent


class RandomAgent(Agent):
    name = 'random_agent'

    def __init__(self):
        pass

    def sample_actions(self, observations, legal_actions) -> int:

        return np.random.choice(legal_actions.reshape(-1).nonzero()[0])

    def update_state(self, state: dict) -> None:
        pass
