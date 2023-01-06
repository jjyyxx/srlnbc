import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

point_goal_config = {
    'robot_base': os.path.join(BASE_DIR, 'xmls/point.xml'),
    'action_scale': [1.0, 0.05],

    'task': 'goal',

    'lidar_num_bins': 16,
    'lidar_alias': True,

    'constrain_hazards': True,
    'constrain_indicator': True,

    'hazards_num': 4,
    'hazards_keepout': 0.4,
    'hazards_size': 0.15,
    'hazards_cost': 1.0,

    'goal_keepout': 0.4,
    'goal_size': 0.3,

    '_seed': None
}

car_goal_config = {
    **point_goal_config,
    'robot_base': 'xmls/car.xml',
    'action_scale': [1.0, 0.02],
}

doggo_goal_config = {
    **point_goal_config,
    'robot_base': 'xmls/doggo.xml',
    'action_scale': 1.0,
    'hazards_num': 2,
    'hazards_size': 0.1,
    'hazards_keepout': 0.5,
    'goal_keepout': 0.5,
    'sensors_obs': 
        ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'] +
        [
            'touch_ankle_1a', 'touch_ankle_2a', 
            'touch_ankle_3a', 'touch_ankle_4a',
            'touch_ankle_1b', 'touch_ankle_2b', 
            'touch_ankle_3b', 'touch_ankle_4b'
        ]
}

point_goal_config_simple = {
    'robot_base': os.path.join(BASE_DIR, 'xmls/point.xml'),
    'action_scale': [1.0, 0.05],

    'task': 'goal',

    'lidar_num_bins': 16,
    'lidar_alias': False,

    'constrain_hazards': True,
    'constrain_indicator': False,

    'hazards_num': 1,
    'hazards_keepout': 1.1,
    'hazards_size': 1.0,
    'hazards_cost': 3.0,
    'hazards_locations': [[0.0, 0.0]],

    'robot_keepout': 0.1,
    'goal_keepout': 0.3,
    'goal_size': 0.2,

    # SimpleEngine specific
    'observe_hazards_pos': True,

    '_seed': None
}
