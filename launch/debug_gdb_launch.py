import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # PPG-SLAM node with GDB debugging
    ppg_slam_debug_node = launch_ros.actions.Node(
        package="ppg_slam",
        executable="ppg_slam_node",
        namespace="ppg_slam", 
        name="ppg_slam_node",
        output="screen",
        parameters=[{
            "vocabulary": 'install/ppg_slam/share/ppg_slam/Vocabulary/voc_euroc_9x3.gz',
            "config": 'install/ppg_slam/share/ppg_slam/config/EuRoC.yaml', 
            "net": 'install/ppg_slam/share/ppg_slam/net'
        }],
        arguments=['--ros-args', '--log-level', 'DEBUG'],
        # Use GDB for debugging
        prefix=['gdb', '-ex', 'run', '--args'],
        emulate_tty=True
    )

    return launch.LaunchDescription([
        ppg_slam_debug_node
    ])
