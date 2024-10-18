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

    method = "statistical_filtering"

    print(f"points {points}")

    start_time = datetime.datetime.now()
    result_cpp = outlier_filter.pc_outlier_filtering(
        las.x, las.y, las.z, method
    )
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

    if method == "statistical_filtering":
        transposed_points = np.transpose(points)

        scipy_start = datetime.datetime.now()
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

        scipy_end = datetime.datetime.now()
        scipy_duration = scipy_end - scipy_start

        print(f"dist_thresh {dist_thresh}")
        print(f"neighbors_distances {neighbors_distances.shape}")
        print(f"scipy time {scipy_duration.seconds}s")

        print(f"scipy time {scipy_duration.microseconds/1000}ms")
        print(points_to_remove)

        # flatten points_to_remove
        detected_points = []
        for removed_point in points_to_remove:
            detected_points.extend(removed_point)

        # print (f"detected_points {detected_points}")

        is_same_result = detected_points == result_cpp
        print(f"Scipy and cars filter resulte are the same ? {is_same_result}")
    elif method == "small_components_filtering":
        transposed_points = np.transpose(points)

        connection_val = 3
        clusters_distance_threshold = None
        nb_pts_threshold = 15

        # mimic what is in outlier_removing_tools
        cloud_tree = cKDTree(transposed_points, leafsize=16)

        # extract connected components
        processed = [False] * len(transposed_points)
        connected_components = []
        for idx, xyz_point in enumerate(transposed_points):
            # if point has already been added to a cluster
            if processed[idx]:
                continue

            # get point neighbors
            neighbors_list = cloud_tree.query_ball_point(
                xyz_point, connection_val
            )

            # add them to the current cluster
            seed = []
            seed.extend(neighbors_list)
            for neigh_idx in neighbors_list:
                processed[neigh_idx] = True

            # iteratively add all the neighbors of the points
            # which were added to the current cluster (if there are some)
            while len(neighbors_list) != 0:
                all_neighbors = cloud_tree.query_ball_point(
                    transposed_points[neighbors_list], connection_val
                )

                # flatten neighbors
                new_neighbors = []
                for neighbor_item in all_neighbors:
                    new_neighbors.extend(neighbor_item)

                # retrieve only new neighbors
                neighbors_list = list(set(new_neighbors) - set(seed))

                # add them to the current cluster
                seed.extend(neighbors_list)
                for neigh_idx in neighbors_list:
                    processed[neigh_idx] = True

            print(f"seed size py {len(seed)}")

            connected_components.append(seed)

        # determine clusters to remove
        cluster_to_remove = []
        for _, connected_components_item in enumerate(connected_components):
            if len(connected_components_item) < nb_pts_threshold:
                if clusters_distance_threshold is not None:
                    # search if the current cluster has any neighbors
                    # in the clusters_distance_threshold radius
                    print("TODO")
                    # all_neighbors = cloud_tree.query_ball_point(
                    #     cloud_xyz[connected_components_item],
                    #     clusters_distance_threshold,
                    # )

                    # # flatten neighbors
                    # new_neighbors = []
                    # for neighbor_item in all_neighbors:
                    #     new_neighbors.extend(neighbor_item)

                    # # retrieve only new neighbors
                    # neighbors_list = list(
                    #     set(new_neighbors) - set(connected_components_item)
                    # )

                    # # if there are no new neighbors, the cluster will be
                    # # removed
                    # if len(neighbors_list) == 0:
                    #     cluster_to_remove.extend(connected_components_item)
                else:
                    cluster_to_remove.extend(connected_components_item)

        print(f"len cluster_to_remove {len(cluster_to_remove)}")
        print(f"cluster_to_remove {cluster_to_remove}")


def console_script():
    """Console script for filterlaz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_in")
    args = parser.parse_args()
    main(args.cloud_in)


if __name__ == "__main__":
    sys.exit(console_script())  # pragma: no cover
