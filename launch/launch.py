import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="rviz2",
                namespace="ppgslam_vis",
                executable="rviz2",
                name="rviz_node",
                output="screen",
                arguments=["-d", PathJoinSubstitution([
                FindPackageShare("ppg_slam"), 
                "rviz", 
                "visualizing.rviz"
                ]).perform(launch.LaunchContext())],
            ),
            launch_ros.actions.Node(
                package="ppg_slam",
                executable="ppg_slam_node",
                namespace="ppg_slam",
                name="ppg_slam_node",
                output="screen",
                parameters=[{
                "vocabulary": '',
                "config": '',
                "net": ''}]
            )
        ]
    )
