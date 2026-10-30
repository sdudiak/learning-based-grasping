from glob import glob
from setuptools import setup, find_packages

name = 'isaac_manipulation_pointcloud'
setup(name=name, version='0.1.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/' + name]),
                  ('share/' + name, ['package.xml'])] +
                 [('share/' + name + '/' + folder, glob(folder + '/' + pattern))
                  for folder, pattern in (('config', '*.yaml'), ('launch', '*.launch.py'), ('rviz', '*.rviz'))],
      install_requires=['setuptools', 'numpy', 'PyYAML'], tests_require=['pytest'],
      maintainer='Staszek', maintainer_email='sdudiak01@gmail.com',
      description='Timestamped visible target and environment point clouds.', license='Apache-2.0',
      entry_points={'console_scripts': ['pointcloud = isaac_manipulation_pointcloud.node:main']})
