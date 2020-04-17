from setuptools import setup

setup(
    name='morphr',
    version='0.1.0',
    packages=[
        'morphr',
        'morphr.constraints',
        'morphr.tasks',
        'morphr.tests',
    ],
    author='Thomas Oberbichler',
    author_email='thomas.oberbichler@tum.de',
    requires=[
        'anurbs',
        'colorama',
        'eqlib',
        'hyperjet',
        'json',
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
