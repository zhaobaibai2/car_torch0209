from setuptools import find_packages, setup

package_name = 'car_iwr_ensemble_torch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['car_iwr_ensemble_torch/config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='IWR-Ensemble for real car',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'car_iwr_ensemble_torch = car_iwr_ensemble_torch.car_iwr_ensemble_torch:main',
            'car_iwr_ensemble_torch_eval_det = car_iwr_ensemble_torch.car_iwr_ensemble_torch_eval:main_det',
            'car_iwr_ensemble_torch_eval_noisy = car_iwr_ensemble_torch.car_iwr_ensemble_torch_eval:main_noisy',
        ],
    },
)
