# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from setuptools import find_packages, setup


VERSION = "0.0.1"

setup(
    name="sonicmoe",
    version=VERSION,
    author="Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao",
    url="",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx", "**/*.yml"]},
)
