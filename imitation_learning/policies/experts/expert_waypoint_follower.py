from .waypoint_follower.waypoint_follow import PurePursuitPlanner
from .expert_base import ExpertBase

class ExpertWaypointFollower(ExpertBase):
    """"
    The objective is to provide an expert driving policy for the F1TENTH car 
    that follows a sequence of waypoints using the Pure Pursuit algorithm. 
    It acts as a wrapper around the PurePursuitPlanner, initializing it with 
    the map configuration and the car's wheelbase, and exposes a plan method 
    that returns the expert's speed and steering angle for a given pose.

    Attributes:
        conf (dict): Configuration for the map, which includes waypoints and other parameters.
        planner (PurePursuitPlanner): An instance of the PurePursuitPlanner initialized with 
            the map configuration and wheelbase.
    """
    def __init__(self, conf):
        if conf is None:
            Exception("map config cannot be None")
        else:
            # The value 0.17145+0.15875 is the wheelbase 
            # (distance between the front and rear axles) of the F1TENTH car, in meters.
            self.planner = PurePursuitPlanner(conf, 0.17145+0.15875)
    
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """it returns the speed and steering angle for the given pose"""
        return self.planner.plan(pose_x, pose_y, pose_theta, lookahead_distance, vgain)

    