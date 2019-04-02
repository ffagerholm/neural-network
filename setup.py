#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=6.0', 'numpy>=1.16.2']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Fredrik Fagerholm",
    author_email='audreyr@example.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="An implementation of a basic feedforward neural network",
    entry_points={
        'console_scripts': [
            'neural_network=neural_network.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='neural_network',
    name='neural_network',
    packages=find_packages(include=['neural_network']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ffagerholm/neural_network',
    version='0.1.0',
    zip_safe=False,
)
