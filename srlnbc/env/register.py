import ray.tune


def register_simple_control():
    from srlnbc.env.car_brake import CarBrake
    from srlnbc.env.unicycle import UnicycleEnv
    from gym.wrappers import TimeLimit

    def car_brake(config):
        return TimeLimit(CarBrake(), 200)

    def unicycle(config):
        return UnicycleEnv()

    ray.tune.registry.register_env('car_brake', car_brake)
    ray.tune.registry.register_env('unicycle', unicycle)


def register_safety_gym():
    from srlnbc.env.config import point_goal_config, car_goal_config, doggo_goal_config
    from srlnbc.env.simple_safety_gym import SimpleEngine

    def point_goal(config):
        return SimpleEngine({**point_goal_config, **config})

    def car_goal(config):
        return SimpleEngine({**car_goal_config, **config})

    def doggo_goal(config):
        return SimpleEngine({**doggo_goal_config, **config})

    ray.tune.registry.register_env('point_goal', point_goal)
    ray.tune.registry.register_env('car_goal', car_goal)
    ray.tune.registry.register_env('doggo_goal', doggo_goal)

def register_metadrive():
    from srlnbc.env.my_metadrive import MySafeMetaDriveEnv
    from srlnbc.env.metadrive_intersection import SafeMetaDriveIntersectionEnv

    def metadrive(config):
        return MySafeMetaDriveEnv({
            "start_seed": 1000,
            "environment_num": 10,
            **config
        })
    
    def metadrive_intersection(config):
        return SafeMetaDriveIntersectionEnv({
            "start_seed": 1000,
            "environment_num": 10,
            **config
        })

    ray.tune.registry.register_env('metadrive', metadrive)
    ray.tune.registry.register_env('metadrive_intersection', metadrive_intersection)
