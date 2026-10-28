from glob import glob

from setuptools import find_packages, setup


package_name = 'isaac_manipulation_segmentation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'PyYAML', 'grasping-segmentation'],
    zip_safe=True,
    maintainer='Staszek',
    maintainer_email='sdudiak01@gmail.com',
    description='Live Mask R-CNN instance segmentation for Isaac Sim camera images.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'segmentation = isaac_manipulation_segmentation.node:main',
    ]},
)
