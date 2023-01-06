"""For evauation."""

from collections import defaultdict
import os
from pathlib import Path
import csv

from metadrive.constants import TerminationState
from srlnbc.agents.safety.mixin import Postprocessing

from srlnbc.env.my_metadrive import MySafeMetaDriveEnv
from srlnbc.env.metadrive_intersection import SafeMetaDriveIntersectionEnv


SINGLETON_ENV = None
def env_creator(config):
    global SINGLETON_ENV
    if SINGLETON_ENV is None:
        SINGLETON_ENV = EvalSafeMetaDriveEnv({
            "start_seed": 1000,
            "environment_num": 10,
            **config,
            "horizon": 1000,
            "record_episode": True
        })
    return SINGLETON_ENV

INTERSECTION_SINGLETON_ENV = None
def intersection_env_creator(config):
    global INTERSECTION_SINGLETON_ENV
    if INTERSECTION_SINGLETON_ENV is None:
        INTERSECTION_SINGLETON_ENV = SafeMetaDriveIntersectionEnv({
            "start_seed": 1000,
            "environment_num": 10,
            **config,
            "horizon": 1000,
            "record_episode": True
        })
    return INTERSECTION_SINGLETON_ENV

class EvalSafeMetaDriveEnv(MySafeMetaDriveEnv):
    def reset(self, *args, **kwargs):
        self.violation_info = defaultdict(bool)
        return super().reset(*args, **kwargs)

    @property
    def violated(self):
        return (
            self.violation_info[TerminationState.CRASH_VEHICLE] or
            self.violation_info[TerminationState.CRASH_OBJECT] or
            self.violation_info[TerminationState.OUT_OF_ROAD] or
            self.violation_info[TerminationState.MAX_STEP]
        )

    def step(self, action):
        obs, rew, done, info = super().step(action)

        self.violation_info[TerminationState.CRASH_VEHICLE] |= info[TerminationState.CRASH_VEHICLE]
        self.violation_info[TerminationState.CRASH_OBJECT] |= info[TerminationState.CRASH_OBJECT]
        self.violation_info[TerminationState.OUT_OF_ROAD] |= info[TerminationState.OUT_OF_ROAD]
        self.violation_info[TerminationState.MAX_STEP] |= info[TerminationState.MAX_STEP]
        self.violation_info[Postprocessing.FEASIBLE] |= info[Postprocessing.FEASIBLE]
        self.violation_info[Postprocessing.INFEASIBLE] |= info[Postprocessing.INFEASIBLE]

        if done and self.violated:
            self.dump_episode()

        return obs, rew, done, info
    
    def dump_episode(self):
        folder = Path("./failures")
        folder.mkdir(exist_ok=True, parents=True)
        info = folder / "_info.csv"
        if not info.exists():
            with info.open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["i", "map", "record", "cv", "co", "oor", "ms", "feas", "infs", "ckpt"])
        with info.open("r") as fr:
            reader = csv.reader(fr)
            next(reader)
            i = 0
            for row in reader:
                i = int(row[0]) + 1
        i = f"{i:03d}"

        filename = f"{i}.pkl"
        self.engine.dump_episode(folder / filename)

        with info.open("a") as fa:
            writer = csv.writer(fa)
            writer.writerow([
                i, self.engine.global_random_seed, filename,
                f"{self.violation_info[TerminationState.CRASH_VEHICLE]!s:>5}",
                f"{self.violation_info[TerminationState.CRASH_OBJECT]!s:>5}",
                f"{self.violation_info[TerminationState.OUT_OF_ROAD]!s:>5}",
                f"{self.violation_info[TerminationState.MAX_STEP]!s:>5}",
                f"{self.violation_info[Postprocessing.FEASIBLE]!s:>5}",
                f"{self.violation_info[Postprocessing.INFEASIBLE]!s:>5}",
                Path(os.environ["CHECKPOINT"]).parent.parent.name
                ])

    def seed(self, *args, **kwargs):
        return super().seed(*args, **kwargs)
