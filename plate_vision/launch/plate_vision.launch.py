"""
카메라 2대로 두 노드를 한 번에 실행

  카메라 A (회전 원판 위): plate_detect_node  - 빈 그릇 / 음식 그릇 구분
  카메라 B (박스 쪽)     : band_count_node    - 파란 띠 개수 → target_count 면 주행로봇 호출

  ros2 launch plate_vision plate_vision.launch.py plate_camera:=0 box_camera:=2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plate_camera', default_value='0', description='카메라 A /dev/videoN (회전 원판)'),
        DeclareLaunchArgument('box_camera', default_value='2', description='카메라 B /dev/videoN (박스)'),
        DeclareLaunchArgument('target_count', default_value='3', description='띠가 이만큼이면 주행로봇 호출'),

        Node(
            package='plate_vision',
            executable='plate_detect_node',
            name='plate_detect_node',
            output='screen',
            parameters=[{'camera_index': LaunchConfiguration('plate_camera')}],
        ),
        Node(
            package='plate_vision',
            executable='band_count_node',
            name='band_count_node',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('box_camera'),
                'target_count': LaunchConfiguration('target_count'),
            }],
        ),
    ])
