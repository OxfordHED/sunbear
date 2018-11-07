import os
from setuptools import setup, find_packages
from sunbear.version import get_version

setup(
    name='sunbear',
    version=get_version(),
    description='Fast Monge-Ampere solver for n-dimensional arrays',
    url='https://github.com/OxfordHED/sunbear',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.14.3",
        "scipy>=1.1.0",
        "matplotlib>=2.2.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6"
    ],
    keywords="optimal-transport",
    zip_safe=False
)
