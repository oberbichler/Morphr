from setuptools import setup

import os
import re


def get_version(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as file:
        data = file.read()
    regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    version_match = re.search(regex, data, re.M)
    if version_match is None:
        raise RuntimeError("Unable to find version string.")
    return version_match.group(1)


setup(
    name='morphr',
    version=get_version(os.path.join('morphr', '__init__.py')),
    packages=[
        'morphr',
        'morphr.objectives',
        'morphr.tasks',
        'morphr.tests',
    ],
    author='Thomas Oberbichler',
    author_email='thomas.oberbichler@tum.de',
    install_requires=[
        'anurbs==0.14.0',
        'colorama==0.4.3',
        'click==7.1.2',
        'eqlib==0.37.1',
        'hyperjet==0.35.0',
        'meshio==4.0.16',
        'numpy==1.19.0',
        'pydantic==1.5.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'morphr=morphr.cli:cli'
        ]
    }
)
