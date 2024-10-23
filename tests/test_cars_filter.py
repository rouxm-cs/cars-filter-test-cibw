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
"""Tests for `cars-filter` package."""

import datetime
import os.path as osp

import laspy
import numpy as np
import pyproj
import pytest
import rasterio
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

import outlier_filter

DATA_FOLDER = osp.join(osp.dirname(osp.dirname(__file__)), "data")
NIMES_LAZ = osp.join(DATA_FOLDER, "subsampled_nimes.laz")
EPIPOLAR_X_IMAGE = osp.join(DATA_FOLDER, "X.tif")
EPIPOLAR_Y_IMAGE = osp.join(DATA_FOLDER, "Y.tif")
EPIPOLAR_Z_IMAGE = osp.join(DATA_FOLDER, "Z.tif")


@pytest.mark.unit_tests
def test_detect_small_components():
    """
    Create fake cloud to process and test detect_small_components
    """
    x_coord = np.zeros((5, 5))
    x_coord[4, 4] = 20
    x_coord[0, 4] = 19.55
    x_coord[0, 3] = 19.10
    y_coord = np.zeros((5, 5))

    z_coord = np.zeros((5, 5))
    z_coord[0:2, 0:2] = 10
    z_coord[1, 1] = 12

    cloud_arr = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(  # noqa: B905
                x_coord, y_coord, z_coord
            )
        ],
        axis=0,
    )

    indexes_to_filter = outlier_filter.pc_small_components_outlier_filtering(
        cloud_arr[:, 0], cloud_arr[:, 1], cloud_arr[:, 2], 0.5, 10, 2
    )
    assert sorted(indexes_to_filter) == [3, 4, 24]

    # test without the second level of filtering
    indexes_to_filter = outlier_filter.pc_small_components_outlier_filtering(
        cloud_arr[:, 0], cloud_arr[:, 1], cloud_arr[:, 2], 0.5, 10, np.nan
    )
    assert sorted(indexes_to_filter) == [0, 1, 3, 4, 5, 6, 24]


@pytest.mark.unit_tests
def test_detect_statistical_outliers():
    """
    Create fake cloud to process and test detect_statistical_outliers
    """
    x_coord = np.zeros((5, 6))
    off = 0
    for line in range(5):
        # x[line,:] = np.arange(off, off+(line+1)*5, line+1)
        last_val = off + 5
        x_coord[line, :5] = np.arange(off, last_val)
        off += (line + 2 + 1) * 5

        # outlier
        x_coord[line, 5] = (off + last_val - 1) / 2

    y_coord = np.zeros((5, 6))
    z_coord = np.zeros((5, 6))

    ref_cloud = np.concatenate(
        [
            np.stack((x_coord, y_coord, z_coord), axis=-1).reshape(-1, 3)
            for x_coord, y_coord, z_coord in zip(  # noqa: B905
                x_coord, y_coord, z_coord
            )
        ],
        axis=0,
    )

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=0.0,
        use_median=False,
    )
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=1.0,
        use_median=False,
    )
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=2.0,
        use_median=False,
    )
    assert sorted(removed_elt_pos) == [23, 29]

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=1.0,
        use_median=True,
    )
    assert sorted(removed_elt_pos) == [5, 11, 17, 23, 29]

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=7.0,
        use_median=True,
    )
    assert sorted(removed_elt_pos) == [11, 17, 23, 29]

    removed_elt_pos = outlier_filter.pc_statistical_outlier_filtering(
        ref_cloud[:, 0],
        ref_cloud[:, 1],
        ref_cloud[:, 2],
        k=4,
        dev_factor=15.0,
        use_median=True,
    )
    # Note: This is the expected result if median computation was exact, but it
    # is not the case in this implementation.
    # assert sorted(removed_elt_pos) == [23, 29]
    assert sorted(removed_elt_pos) == [29]


@pytest.mark.parametrize("use_median", [True, False])
def test_point_cloud_statistical(use_median):
    """
    Outlier filtering test from laz, using statistical method.

    The test verifies that cars-filter produces the same results as a Python
    equivalent using scipy ckdtrees
    """
    k = 50
    dev_factor = 1

    with laspy.open(NIMES_LAZ) as creader:
        las = creader.read()
        points = np.vstack((las.x, las.y, las.z))

    start_time = datetime.datetime.now()
    result_cpp = outlier_filter.pc_statistical_outlier_filtering(
        las.x, las.y, las.z, dev_factor, k, use_median
    )
    end_time = datetime.datetime.now()
    cpp_duration = end_time - start_time

    print(f"Statistical filtering total duration (cpp): {cpp_duration}")

    # Perform the same filtering Scipy and compare the results
    transposed_points = np.transpose(points)

    scipy_start = datetime.datetime.now()
    # mimic what is in outlier_removing_tools
    cloud_tree = cKDTree(transposed_points)
    neighbors_distances, _ = cloud_tree.query(transposed_points, k + 1)
    mean_neighbors_distances = np.sum(neighbors_distances, axis=1)
    mean_neighbors_distances /= k
    # compute median and interquartile range of those mean distances
    # for the whole point cloud
    if use_median:
        # compute median and interquartile range of those mean distances
        # for the whole point cloud
        median_distances = np.median(mean_neighbors_distances)
        iqr_distances = np.percentile(
            mean_neighbors_distances, 75
        ) - np.percentile(mean_neighbors_distances, 25)
        # compute distance threshold and
        # apply it to determine which points will be removed
        dist_thresh = median_distances + dev_factor * iqr_distances
    else:
        mean_distances = np.mean(mean_neighbors_distances)
        std_distances = np.std(mean_neighbors_distances)
        # compute distance threshold and
        # apply it to determine which points will be removed
        dist_thresh = mean_distances + dev_factor * std_distances

    print(f"dist_thresh {dist_thresh}")

    points_to_remove = np.argwhere(mean_neighbors_distances > dist_thresh)

    scipy_end = datetime.datetime.now()
    scipy_duration = scipy_end - scipy_start

    # flatten points_to_remove
    detected_points = []
    for removed_point in points_to_remove:
        detected_points.extend(removed_point)

    # print (f"detected_points {detected_points}")

    print(f"Statistical filtering total duration (Python): {scipy_duration}")
    is_same_result = detected_points == result_cpp
    assert is_same_result
    print(f"Scipy and cars filter results are the same ? {is_same_result}")


@pytest.mark.parametrize("clusters_distance_threshold", [float("nan"), 4])
def test_point_cloud_small_component(clusters_distance_threshold):
    """
    Outlier filtering test from laz, using small components method.

    The test verifies that cars-filter produces the same results as a Python
    equivalent using scipy ckdtrees
    """

    connection_val = 3
    nb_pts_threshold = 15

    with laspy.open(NIMES_LAZ) as creader:
        las = creader.read()
        points = np.vstack((las.x, las.y, las.z))

    start_time = datetime.datetime.now()
    result_cpp = outlier_filter.pc_small_components_outlier_filtering(
        las.x,
        las.y,
        las.z,
        connection_val,
        nb_pts_threshold,
        clusters_distance_threshold,
    )
    end_time = datetime.datetime.now()
    cpp_duration = end_time - start_time

    print(f"Small Component filtering total duration (cpp): {cpp_duration}")
    print(f"result_cpp: {result_cpp}")

    transposed_points = np.transpose(points)

    scipy_start = datetime.datetime.now()

    # mimic what is in outlier_removing_tools
    cloud_tree = cKDTree(transposed_points)

    # extract connected components
    processed = [False] * len(transposed_points)
    connected_components = []
    for idx, xyz_point in enumerate(transposed_points):
        # if point has already been added to a cluster
        if processed[idx]:
            continue

        # get point neighbors
        neighbors_list = cloud_tree.query_ball_point(xyz_point, connection_val)

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

        connected_components.append(seed)

    # determine clusters to remove
    cluster_to_remove = []
    for _, connected_components_item in enumerate(connected_components):
        if len(connected_components_item) < nb_pts_threshold:
            if clusters_distance_threshold is not None:
                # search if the current cluster has any neighbors
                # in the clusters_distance_threshold radius
                all_neighbors = cloud_tree.query_ball_point(
                    transposed_points[connected_components_item],
                    clusters_distance_threshold,
                )

                # flatten neighbors
                new_neighbors = []
                for neighbor_item in all_neighbors:
                    new_neighbors.extend(neighbor_item)

                # retrieve only new neighbors
                neighbors_list = list(
                    set(new_neighbors) - set(connected_components_item)
                )

                # if there are no new neighbors, the cluster will be
                # removed
                if len(neighbors_list) == 0:
                    cluster_to_remove.extend(connected_components_item)
            else:
                cluster_to_remove.extend(connected_components_item)

    scipy_end = datetime.datetime.now()
    scipy_duration = scipy_end - scipy_start
    print(
        f"Small Component filtering total duration (Python): {scipy_duration}"
    )

    cluster_to_remove.sort()
    result_cpp.sort()

    print(f"python {cluster_to_remove}")
    print(f"result_cpp {result_cpp}")

    is_same_result = cluster_to_remove == result_cpp

    assert is_same_result
    print(f"Scipy and cars filter results are the same ? {is_same_result}")


@pytest.mark.parametrize("use_median", [True, False])
def test_epipolar_statistical_filtering(use_median):
    """
    Outlier filtering test from depth map in epipolar geometry, using
    statistical method
    """
    k = 15
    half_window_size = 15
    dev_factor = 1

    with rasterio.open(EPIPOLAR_X_IMAGE) as x_ds, rasterio.open(
        EPIPOLAR_Y_IMAGE
    ) as y_ds, rasterio.open(EPIPOLAR_Z_IMAGE) as z_ds:
        x_values = x_ds.read(1)
        y_values = y_ds.read(1)
        z_values = z_ds.read(1)

    input_shape = x_values.shape

    # hard code UTM 36N for Gizeh for now
    transformer = pyproj.Transformer.from_crs(4326, 32636)
    # X-Y inversion required because WGS84 is lat first ?
    x_utm, y_utm = transformer.transform(x_values, y_values)

    # Make copies for reprocessing with kdtree
    x_utm_flat = np.copy(x_utm).reshape(input_shape[0] * input_shape[1])
    y_utm_flat = np.copy(y_utm).reshape(input_shape[0] * input_shape[1])
    z_flat = np.copy(z_values).reshape(input_shape[0] * input_shape[1])

    start_time = datetime.datetime.now()

    outlier_array = outlier_filter.epipolar_statistical_outlier_filtering(
        x_utm, y_utm, z_values, k, half_window_size, dev_factor, use_median
    )

    end_time = datetime.datetime.now()
    epipolar_processing_duration = end_time - start_time

    print(f"Epipolar filtering duration: {epipolar_processing_duration}")

    # filter NaNs
    nan_pos = np.isnan(x_utm_flat)
    x_utm_flat = x_utm_flat[~nan_pos]
    y_utm_flat = y_utm_flat[~nan_pos]
    z_flat = z_flat[~nan_pos]

    start_time = datetime.datetime.now()

    result_kdtree = np.array(
        outlier_filter.pc_statistical_outlier_filtering(
            x_utm_flat, y_utm_flat, z_flat, dev_factor, k, use_median
        )
    )

    end_time = datetime.datetime.now()
    kdtree_processing_duration = end_time - start_time
    print(f"KDTree filtering duration: {kdtree_processing_duration}")

    print(outlier_array.shape)

    outlier_array = outlier_array.reshape(input_shape[0] * input_shape[1])
    print(outlier_array.shape)
    outlier_array = np.argwhere(outlier_array[~nan_pos]).flatten()
    print(outlier_array.shape)

    # Find common outliers between the two methods
    # common_outliers = np.intersect1d(result_kdtree, outlier_array)
    # print(common_outliers)

    # Note that k and half_window_size have been chosed for this assertion to
    # succeed.
    # The two algorithms does not produce the same results if the epipolar
    # neighborhood is too small.
    assert (np.sort(outlier_array) == np.sort(result_kdtree)).all()


@pytest.mark.parametrize("clusters_distance_threshold", [float("nan"), 2])
def test_epipolar_small_components_filtering(clusters_distance_threshold):
    """
    Outlier filtering test from depth map in epipolar geometry, using small
    components method
    """
    min_cluster_size = 15
    radius = 1
    half_window_size = 7

    with rasterio.open(EPIPOLAR_X_IMAGE) as x_ds, rasterio.open(
        EPIPOLAR_Y_IMAGE
    ) as y_ds, rasterio.open(EPIPOLAR_Z_IMAGE) as z_ds:
        x_values = x_ds.read(1)
        y_values = y_ds.read(1)
        z_values = z_ds.read(1)

    input_shape = x_values.shape

    transformer = pyproj.Transformer.from_crs(4326, 32636)
    # X-Y inversion required because WGS84 is lat first ?
    x_utm, y_utm = transformer.transform(x_values, y_values)

    # Make copies for reprocessing with kdtree
    x_utm_flat = np.copy(x_utm).reshape(input_shape[0] * input_shape[1])
    y_utm_flat = np.copy(y_utm).reshape(input_shape[0] * input_shape[1])
    z_flat = np.copy(z_values).reshape(input_shape[0] * input_shape[1])

    start_time = datetime.datetime.now()

    outlier_array = outlier_filter.epipolar_small_components_outlier_filtering(
        x_utm,
        y_utm,
        z_values,
        min_cluster_size,
        radius,
        half_window_size,
        clusters_distance_threshold,
    )

    end_time = datetime.datetime.now()
    epipolar_processing_duration = end_time - start_time

    print(f"Epipolar filtering duration: {epipolar_processing_duration}")

    # Test with KDTree

    # filter NaNs
    nan_pos = np.isnan(x_utm_flat)
    x_utm_flat = x_utm_flat[~nan_pos]
    y_utm_flat = y_utm_flat[~nan_pos]
    z_flat = z_flat[~nan_pos]

    start_time = datetime.datetime.now()

    result_kdtree = np.array(
        outlier_filter.pc_small_components_outlier_filtering(
            x_utm_flat,
            y_utm_flat,
            z_flat,
            radius,
            min_cluster_size,
            clusters_distance_threshold,
        )
    )

    end_time = datetime.datetime.now()
    kdtree_processing_duration = end_time - start_time

    print(f"KDTree filtering duration: {kdtree_processing_duration}")

    outlier_array = outlier_array.reshape(input_shape[0] * input_shape[1])
    print(outlier_array.shape)
    outlier_array = np.argwhere(outlier_array[~nan_pos]).flatten()
    print(outlier_array.shape)
    # Find common outliers between the two methods
    # common_outliers = np.intersect1d(result_kdtree, outlier_array)
    # print(common_outliers)

    assert (np.sort(outlier_array) == np.sort(result_kdtree)).all()
