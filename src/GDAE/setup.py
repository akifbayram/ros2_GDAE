from setuptools import setup
from glob import glob
import os

package_name = 'gdae'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    py_modules=[
        'script.gdae_tb4',
        'script.gdae_tb3'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mehmet Bayram',
    maintainer_email='akif@example.com',
    description='Goal Driven Autonomous Exploration',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gdae_tb4 = script.gdae_tb4:main',
        ],
    },
)