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


def main(**kwargs):
    """
    Filter LAS
    """

    with laspy.open(kwargs["cloud_in"]) as creader:
        las = creader.read()

    method = kwargs["method"]

    start_time = datetime.datetime.now()

    if method == "statistical":
        outliers = outlier_filter.pc_statistical_outlier_filtering(
            las.x,
            las.y,
            las.z,
            k=kwargs["statistical.k"],
            dev_factor=kwargs["statistical.dev_factor"],
            use_median=kwargs["statistical.use_median"],
        )
    elif method == "small_components":
        outliers = outlier_filter.pc_small_components_outlier_filtering(
            las.x,
            las.y,
            las.z,
            radius=kwargs["small_components.radius"],
            min_cluster_size=kwargs["small_components.min_cluster_size"],
            clusters_distance_threshold=kwargs[
                "small_components.clusters_distance_threshold"
            ],
        )
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print(f"pc_outlier_filtering duration (without IO): {duration}")

    # Write results to file
    valid_file = laspy.create(
        point_format=las.header.point_format, file_version=las.header.version
    )
    valid_indices = [i for i in range(len(las)) if i not in outliers]
    valid_file.points = las.points[valid_indices]

    valid_file.write(kwargs["cloud_out_valid"])


def console_script():
    """Console script for filterlaz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_in")
    parser.add_argument("cloud_out_valid")

    subparsers = parser.add_subparsers(help="filtering method", dest="method")

    statistical_parser = subparsers.add_parser("statistical")
    small_components_parser = subparsers.add_parser("small_components")

    # Statistical filtering parameters
    statistical_parser.add_argument(
        "statistical.k", nargs="?", type=int, default=50
    )
    statistical_parser.add_argument(
        "statistical.dev_factor", nargs="?", type=float, default=1
    )
    statistical_parser.add_argument(
        "statistical.use_median", nargs="?", type=bool, default=False
    )

    # Small components parameters
    small_components_parser.add_argument(
        "small_components.radius", nargs="?", type=float, default=3.0
    )
    small_components_parser.add_argument(
        "small_components.min_cluster_size", nargs="?", type=int, default=15
    )
    small_components_parser.add_argument(
        "small_components.clusters_distance_threshold",
        nargs="?",
        type=float,
        default=np.nan,
    )

    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    sys.exit(console_script())  # pragma: no cover
