import numpy as np

from simulator.env.table_clearing import EpisodeState
from hparams import read_params


class RewardCalculator(object):

    _s_tm1: EpisodeState

    def __init__(self, initial_state, params=read_params('table_clearing_base')):
        self._s_tm1 = initial_state
        self._params = params

    def update(self, state: EpisodeState):
        reward = 0.0

        # Reaching
        if not state.grasped:
            reward += self._params.reward.reaching * np.sign(self._s_tm1.d_tg - state.d_tg)

        # Delivering
        if state.grasped:
            reward += self._params.reward.delivering * np.sign(self._s_tm1.d_gd - state.d_gd)

        # Successful grasp
        if not self._s_tm1.grasped and state.grasped:
            reward += self._params.reward.grasped

        reached_destination = state.d_gd < 0.05

        # Dropped target
        if self._s_tm1.grasped and not state.grasped and not reached_destination:
            reward += self._params.reward.dropped

        # Reached destination
        if self._s_tm1.grasped and not state.grasped and reached_destination:
            reward += self._params.reward.delivered

        # Collided
        if state.collided:
            reward += self._params.reward.collided

        self._s_tm1 = state

        return reward
