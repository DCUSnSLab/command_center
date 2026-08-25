from setuptools import setup

package_name = 'waypoint_manage'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/waypoint_manage.launch.py']),
        ('share/' + package_name + '/config', ['config/waypoint_manage.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d2-521-30',
    maintainer_email='min81800@cu.ac.kr',
    description='Waypoint layer: semantic target -> guarded odom-frame waypoints',
    license='MIT',
    entry_points={
        'console_scripts': [
            'waypoint_manage_node = waypoint_manage.waypoint_manage_node:main',
        ],
    },
)
