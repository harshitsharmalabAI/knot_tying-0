"""ros2 launch ./knot_live.launch.py [sim_args:="--t-cinch 15"] [stop_on_clash:=true]"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def setup(context):
    urdf = os.path.abspath(LaunchConfiguration("urdf").perform(context))
    sim_dir = os.path.abspath(LaunchConfiguration("sim_dir").perform(context))
    if not os.path.exists(urdf):
        raise RuntimeError(f"{urdf} not found - run dual_arm.py first")
    for f in ("tension_knot.py", "frictional_contact.py", "wound_tissue.py"):
        if not os.path.exists(os.path.join(sim_dir, f)):
            raise RuntimeError(f"{f} not found in {sim_dir} (set sim_dir:=...)")
    cmd = ["python3", os.path.join(HERE, "knot_live.py"), "--urdf", urdf, "--sim-dir", sim_dir,
           "--live-dt", LaunchConfiguration("live_dt").perform(context)]
    if LaunchConfiguration("stop_on_clash").perform(context).lower() in ("1", "true", "yes"):
        cmd.append("--stop-on-clash")
    sim_args = LaunchConfiguration("sim_args").perform(context).split()
    if sim_args:
        cmd += ["--"] + sim_args
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": open(urdf).read()}], output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(HERE, "knot.rviz")],
             output="screen"),
        ExecuteProcess(cmd=cmd, cwd=sim_dir, output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("urdf", default_value=os.path.join(HERE, "dual_ur7e_grasper.urdf")),
        DeclareLaunchArgument("sim_dir", default_value=os.path.dirname(HERE)),
        DeclareLaunchArgument("sim_args", default_value="--t-cinch 15"),
        DeclareLaunchArgument("live_dt", default_value="0.005"),
        DeclareLaunchArgument("stop_on_clash", default_value="false"),
        OpaqueFunction(function=setup),
    ])
