from setuptools import find_packages, setup

package_name = 'kiosk_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joheeho',
    maintainer_email='joheeho@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_pnp_node = kiosk_vision.aruco_pnp_node:main',
            'approach_control_node = kiosk_vision.approach_control_node:main',
        ],
    },
)
