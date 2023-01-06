from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.road import Road
from metadrive.envs.marl_envs.marl_intersection import MAIntersectionMap
from metadrive.utils.config import Config
from metadrive.manager.map_manager import MapManager

from srlnbc.env.my_metadrive import MySafeMetaDriveEnv

IntersectionConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    ],
    map_config=dict(exit_length=60, lane_num=3),
    traffic_density=0.2,
)

class IntersectionMapManager(MapManager):
    """Functionally identical to MAIntersectionMapManager, but fix a memory leak."""
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAIntersectionMap, map_config=config["map_config"], random_seed=None)
            self.spawned_objects[_map.id] = _map
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = next(iter(self.spawned_objects.values()))
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]

class SafeMetaDriveIntersectionEnv(MySafeMetaDriveEnv):
    def default_config(self) -> Config:
        return super().default_config().update(IntersectionConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(SafeMetaDriveIntersectionEnv, self).setup_engine()
        self.engine.update_manager("map_manager", IntersectionMapManager())
