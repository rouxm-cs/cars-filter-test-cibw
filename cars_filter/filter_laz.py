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
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

import outlier_filter


def main(cloud_in):
    """
    Filter LAS
    """
    with laspy.open(cloud_in) as creader:
        las = creader.read()
        print(f"las: {las}")
        points = np.vstack((las.x, las.y, las.z))

    print(f"points {points}")

    start_time = datetime.datetime.now()
    result_cpp = outlier_filter.pc_outlier_filtering(points)
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time

    print(
        "pc_outlier_filtering total duration " + f"{total_duration.seconds}s."
    )
    print(
        "pc_outlier_filtering total duration "
        + f"{total_duration.microseconds/1000}ms."
    )

    # print(f"result cpp {result_cpp}")

    transposed_points = np.transpose(points)

    a = datetime.datetime.now()
    # mimic what is in outlier_removing_tools
    cloud_tree = cKDTree(transposed_points, leafsize=16)
    k = 50
    neighbors_distances, _ = cloud_tree.query(transposed_points, k)
    mean_neighbors_distances = np.sum(neighbors_distances, axis=1)
    mean_neighbors_distances /= k
    # compute median and interquartile range of those mean distances
    # for the whole point cloud
    mean_distances = np.mean(mean_neighbors_distances)
    std_distances = np.std(mean_neighbors_distances)
    # compute distance threshold and
    # apply it to determine which points will be removed
    dist_thresh = mean_distances + 1 * std_distances

    points_to_remove = np.argwhere(mean_neighbors_distances > dist_thresh)

    b = datetime.datetime.now()
    c = b - a

    print(f"dist_thresh {dist_thresh}")
    print(f"neighbors_distances {neighbors_distances.shape}")
    print(f"scipy time {c.seconds}s")

    print(f"scipy time {c.microseconds/1000}ms")
    print(points_to_remove)

    # flatten points_to_remove
    detected_points = []
    for removed_point in points_to_remove:
        detected_points.extend(removed_point)

    # print (f"detected_points {detected_points}")

    is_same_result = detected_points == result_cpp
    print(f"Scipy and cars filter resulte are the same ? {is_same_result}")


def console_script():
    """Console script for filterlaz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_in")
    args = parser.parse_args()
    main(args.cloud_in)


if __name__ == "__main__":
    sys.exit(console_script())  # pragma: no cover
