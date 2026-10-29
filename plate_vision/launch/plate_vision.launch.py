"""
카메라 2대로 두 노드를 한 번에 실행

  ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 band_camera:=2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plate_camera', default_value='0',
                              description='회전 테이블을 내려다보는 카메라 /dev/videoN (빈 접시 판별)'),
        DeclareLaunchArgument('band_camera', default_value='2',
                              description='트레이에 쌓인 접시 옆면을 보는 카메라 /dev/videoN (파란 띠 카운트)'),

        Node(
            package='plate_vision',
            executable='plate_state_node',
            name='plate_state_node',
            output='screen',
            parameters=[{'camera_index': LaunchConfiguration('plate_camera')}],
        ),
        Node(
            package='plate_vision',
            executable='band_count_node',
            name='band_count_node',
            output='screen',
            parameters=[{'camera_index': LaunchConfiguration('band_camera')}],
        ),
    ])
