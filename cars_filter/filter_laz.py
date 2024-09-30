#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of cars-filter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Filter laz utility script"""

import argparse
import datetime
import sys

import laspy
import numpy as np
import outlier_filter


def main(cloud_in):
    """
    Filter LAS
    """
    with laspy.open(cloud_in) as creader:
        las = creader.read()
        points = np.vstack((las.x, las.y, las.z))

    print(f"points {points}")

    start_time = datetime.datetime.now()
    outlier_filter.pc_outlier_filtering(points)
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time

    print(
        "pc_outlier_filtering total duration "
        + f"{total_duration.microseconds/1000}ms."
    )


def console_script():
    """Console script for filterlaz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_in")
    args = parser.parse_args()
    main(args.cloud_in)


if __name__ == "__main__":
    sys.exit(console_script())  # pragma: no cover
