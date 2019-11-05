from comet_ml import Experiment, ExistingExperiment
from comet_ml._logging import _reset_already_imported_modules


def new_experiment():
    return Experiment(
        api_key="lzBiJi5UcZCwhAqlFNCUP4Qpg",
        project_name="athena",
        workspace="skr-io7803",
    )


def wrap_experiment(experiment):
    class CometPickle(object):
        comet: Experiment

        def __init__(self, comet):
            self.comet = comet

        def __getstate__(self):
            return {
                'api_key': self.comet.api_key,
                'experiment_key': self.comet.get_key()
            }

        def __setstate__(self, state):
            api_key = state['api_key']
            exp_key = state['experiment_key']
            _reset_already_imported_modules()
            self.comet = ExistingExperiment(
                api_key=api_key,
                previous_experiment=exp_key,
                log_code=False,
                log_graph=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_output_logging=None,
                log_env_details=False,
                log_git_metadata=False,
                log_git_patch=False,
                disabled=False,
                log_env_gpu=False,
                log_env_host=False,
                display_summary=None,
                log_env_cpu=False
            )

    return CometPickle(experiment)
