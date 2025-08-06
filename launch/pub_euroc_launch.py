import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="ppg_slam",
                namespace="ppg_slam",
                executable="pub_euroc_node",
                name="pub_euroc_node",
                output="screen"
            )
        ]
    )
