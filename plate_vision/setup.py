from glob import glob

from setuptools import setup

package_name = 'plate_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Subin Lee',
    maintainer_email='20233130@gs.cwnu.ac.kr',
    description='Empty/food plate detection and red band counting',
    license='MIT',
    entry_points={
        'console_scripts': [
            'plate_state_node = plate_vision.plate_state_node:main',
            'band_count_node = plate_vision.band_count_node:main',
            'red_count_node = plate_vision.red_count_node:main',
        ],
    },
)
