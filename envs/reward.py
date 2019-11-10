from attrdict import AttrDict, AttrDefault


class RewardLog(object):

    def __init__(self):
        super(RewardLog, self).__init__()
        self._log = AttrDict()

    def _log_item(self, key, value):
        if key not in self._log:
            self._log[key] = AttrDict(
                count=0,
                sum=0,
                mean=0.0,
                max=0.0,
                min=0.0
            )

        log = self._log[key]

        log.count += 1
        log.recent = value
        log.sum += value
        log.mean = log.sum / log.count
        log.max = max(log.max, value)
        log.min = min(log.min, value)

    def log(self, session):
        for k, v in session.items():
            self._log_item(k, v)

    @property
    def info(self):
        return {k: dict(v) for k, v in self._log.items()}


class RewardSession(AttrDict):

    def sum(self):
        return sum(self.values())
