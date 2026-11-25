"""ros2 launch ./knot_replay.launch.py [speed:=0.5] [thread_width:=0.003]"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def setup(context):
    urdf = os.path.abspath(LaunchConfiguration("urdf").perform(context))
    motion = os.path.abspath(LaunchConfiguration("motion").perform(context))
    speed = LaunchConfiguration("speed").perform(context)
    width = LaunchConfiguration("thread_width").perform(context)
    tissue = LaunchConfiguration("tissue_thickness").perform(context)
    for f in (urdf, motion):
        if not os.path.exists(f):
            raise RuntimeError(f"{f} not found - run plan_motion.py first")
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": open(urdf).read()}], output="screen"),
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "knot_replay.py"), "--ros-args",
                            "-p", f"motion:={motion}", "-p", f"speed:={speed}",
                            "-p", f"thread_width:={width}",
                            "-p", f"tissue_thickness:={tissue}"], output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(HERE, "knot.rviz")],
             output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("urdf", default_value="dual_ur7e_grasper.urdf"),
        DeclareLaunchArgument("motion", default_value="knot_motion.npz"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("thread_width", default_value="0.0015", description="drawn thread width (m)"),
        DeclareLaunchArgument("tissue_thickness", default_value="0.015", description="drawn pad thickness (m)"),
        OpaqueFunction(function=setup),
    ])
