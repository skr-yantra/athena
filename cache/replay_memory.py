import pandas as pd
import numpy as np

_COLUMNS = [
    'episode_id',
    's_t',
    'a_t',
    'r_t'
]


class ReplayMemory:

    def __init__(self, max_size):
        self._max_size = max_size
        self._data = pd.DataFrame(columns=_COLUMNS)

    def post_state(self, episode_id, env_state, action, reward):
        self._data[len(self._data)] = [episode_id, env_state, action, reward]

        if len(self._data) > self._max_size:
            self._data.drop(0, axis=0, inplace=True)

    def generate_batch(self, batch_size, gamma):
        samples = self._data.sample(n=batch_size)

        def _reward_calculator(row):
            start = row.name
            end = self._data.index[self._data.episode == row.episode].max()

            return np.sum(self._data.r_t.loc[start:end + 1] * (gamma ** np.arange(end - start + 1)))

        samples.r_t = samples.apply(_reward_calculator, axis=1)

        return samples
