from setuptools import setup
import os
from glob import glob

package_name = 'simple_costmap_2d'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='D2-521-30',
    maintainer_email='user@todo.todo',
    description='Simple 2D costmap implementation for robot navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'costmap_node = simple_costmap_2d.costmap_node:main',
        ],
    },
)
