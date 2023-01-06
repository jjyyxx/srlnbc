import os
from typing import Dict, List

import ray.tune, ray.tune.progress_reporter
from ray.tune.logger import (
    LoggerCallback,
    # CSVLoggerCallback,
    JsonLoggerCallback,
    TBXLoggerCallback,
)

class JsonLoggerCallbackNoHist(JsonLoggerCallback):
    # Include this logger to be compatible with ExperimentAnalysis tool
    # Also this will dump params.{pkl,json}
    def log_trial_result(self, iteration: int, trial, result: Dict):
        result = result.copy()
        if "hist_stats" in result:
            del result["hist_stats"]
        return super().log_trial_result(iteration, trial, result)

def create_progress():
    return (
        ray.tune.JupyterNotebookReporter(overwrite=True, print_intermediate_tables=True)
        if ray.tune.progress_reporter.IS_NOTEBOOK else
        ray.tune.CLIReporter(print_intermediate_tables=True)
    )

def create_callbacks() -> List[LoggerCallback]:
    # To avoid creating CSVLoggerCallback and JsonLoggerCallback
    # which leads to bloated log files due to hist_data
    # Use with
    #   ray.tune.run(..., callbacks=create_callbacks(), ...)
    os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    return [
        JsonLoggerCallbackNoHist(),
        TBXLoggerCallback(),
    ]
