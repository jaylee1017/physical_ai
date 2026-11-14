"""
카메라 1대 + YOLO 로 빈 그릇을 세고, target_count 개가 되면 주행로봇 호출

  ros2 launch plate_vision empty_count.launch.py camera:=0 target_count:=3
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='0', description='위에서 보는 카메라 /dev/videoN'),
        DeclareLaunchArgument('target_count', default_value='3', description='이만큼 빈 그릇이 모이면 주행로봇 호출'),

        Node(
            package='plate_vision',
            executable='empty_count_node',
            name='empty_count_node',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('camera'),
                'target_count': LaunchConfiguration('target_count'),
            }],
        ),
    ])
