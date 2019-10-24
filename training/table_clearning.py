from simulator.env.table_clearing import EpisodeState


class RewardCalculator(object):

    _s_tm1: EpisodeState

    def __init__(self):
        self._s_tm1 = None

    def update(self, state):
        # TODO
        pass
