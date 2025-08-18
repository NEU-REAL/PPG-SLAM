import launch
import launch_ros.actions
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Declare launch arguments
    vocabulary_arg = DeclareLaunchArgument(
        'vocabulary',
        default_value='install/ppg_slam/share/ppg_slam/Vocabulary/voc_euroc_9x3.gz',
        description='Path to vocabulary file'
    )
    
    config_arg = DeclareLaunchArgument(
        'config', 
        default_value='install/ppg_slam/share/ppg_slam/config/EuRoC.yaml',
        description='Path to config file'
    )
    
    net_arg = DeclareLaunchArgument(
        'net',
        default_value='install/ppg_slam/share/ppg_slam/net',
        description='Path to neural network files'
    )
    
    use_gdb_arg = DeclareLaunchArgument(
        'use_gdb',
        default_value='false',
        description='Whether to use gdb for debugging'
    )
    
    use_valgrind_arg = DeclareLaunchArgument(
        'use_valgrind', 
        default_value='false',
        description='Whether to use valgrind for memory debugging'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Log level (DEBUG, INFO, WARN, ERROR, FATAL)'
    )

    # PPG-SLAM node
    ppg_slam_node = launch_ros.actions.Node(
        package="ppg_slam",
        executable="ppg_slam_node", 
        namespace="ppg_slam",
        name="ppg_slam_node",
        output="screen",
        parameters=[{
            "vocabulary": LaunchConfiguration('vocabulary'),
            "config": LaunchConfiguration('config'),
            "net": LaunchConfiguration('net')
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        # Enable debugging with gdb
        prefix=['gdb -ex run --args'] if LaunchConfiguration('use_gdb').perform(launch.LaunchContext()) == 'true' else None,
        # Enable memory debugging with valgrind  
        # prefix=['valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose'] if LaunchConfiguration('use_valgrind').perform(launch.LaunchContext()) == 'true' else None,
        emulate_tty=True
    )
    
    # RViz node for visualization
    rviz_node = launch_ros.actions.Node(
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
    )

    return launch.LaunchDescription([
        vocabulary_arg,
        config_arg, 
        net_arg,
        use_gdb_arg,
        use_valgrind_arg,
        log_level_arg,
        ppg_slam_node,
        rviz_node
    ])
