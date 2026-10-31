"""
위에서 보는 카메라 1대로 빈 접시 판별 + 트레이 파란 띠 세기

  ros2 launch plate_vision top_vision.launch.py camera:=0 band_mode:=blobs
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='0', description='위에서 보는 카메라 /dev/videoN'),
        DeclareLaunchArgument('band_mode', default_value='blobs',
                              description='blobs: 트레이에 나란히 / events: 겹쳐 쌓기'),
        DeclareLaunchArgument('target_count', default_value='3'),

        Node(
            package='plate_vision',
            executable='top_vision_node',
            name='top_vision_node',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('camera'),
                'band_mode': LaunchConfiguration('band_mode'),
                'target_count': LaunchConfiguration('target_count'),
            }],
        ),
    ])
