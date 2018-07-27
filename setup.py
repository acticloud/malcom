from setuptools import setup, find_packages

setup(
    name="malcom",
    version="0.2-dev",
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
