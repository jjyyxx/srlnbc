from typing import Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class SafetyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode, **kwargs) -> None:
        episode.user_data["total_cost"] = 0

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]] = None, episode: Episode, **kwargs) -> None:
        info = episode.last_info_for()
        episode.user_data["total_cost"] += float(info['cost'] > 0)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                    policies: Dict[str, Policy], episode: Episode,
                    env_index: int, **kwargs):
        episode.custom_metrics["episode_cost"] = episode.user_data["total_cost"]
        if "penalty_margin" in episode.user_data:
            episode.hist_data["penalty_margin"] = episode.user_data["penalty_margin"].tolist()
        info = episode.last_info_for()
        episode.custom_metrics["cost_exception"] = info.get("cost_exception", 0.0)
