from setuptools import find_packages, setup

package_name = 'execute_trajectories'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jala',
    maintainer_email='jala@universal-robots.com',
    description='SDU Summer school starting script',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur_trajectory_with_data_recording = execute_trajectories.ur_trajectory_with_data_recording:main',
        ],
    },
)
