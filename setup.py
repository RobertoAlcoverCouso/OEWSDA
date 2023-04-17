from setuptools import find_packages
from setuptools import setup

setup(name='OEWSSSS',
      install_requires=['pyyaml', 'tensorboardX',
                        'easydict', 'matplotlib',
                        'scipy', 'scikit-image',
                        'future', 'setuptools',
                        'tqdm', 'cffi',
                        'pandas'],
      packages=find_packages())
