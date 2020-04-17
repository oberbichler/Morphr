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
        'morphr.constraints',
        'morphr.tasks',
        'morphr.tests',
    ],
    author='Thomas Oberbichler',
    author_email='thomas.oberbichler@tum.de',
    install_requires=[
        'anurbs',
        'colorama',
        'eqlib',
        'hyperjet',
        'meshio',
        'numpy',
        'pydantic',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'morphr=morphr.cli:main'
        ]
    }
)
