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
"""Filter epipolar utility script"""

import argparse
import datetime
import sys

import numpy as np
import pyproj
import rasterio

import outlier_filter


def main(**kwargs):
    """
    Filter outliers in epipolar images
    """

    # Parse common parameters
    in_epipolar_x_image = kwargs["in_epipolar_x_image"]
    in_epipolar_y_image = kwargs["in_epipolar_y_image"]
    in_epipolar_z_image = kwargs["in_epipolar_z_image"]

    outlier_mask = kwargs["out_outliers"]
    out_epipolar_x_image = kwargs["out_epipolar_x_image"]
    out_epipolar_y_image = kwargs["out_epipolar_y_image"]
    out_epipolar_z_image = kwargs["out_epipolar_z_image"]

    epsg = kwargs["epsg"]
    method = kwargs["method"]

    with rasterio.open(in_epipolar_x_image) as x_ds, rasterio.open(
        in_epipolar_y_image
    ) as y_ds, rasterio.open(in_epipolar_z_image) as z_ds:
        x_values = x_ds.read(1)
        y_values = y_ds.read(1)
        z_values = z_ds.read(1)

        transformer = pyproj.Transformer.from_crs(4326, epsg)
        # X-Y inversion required because WGS84 is lat first ?
        # pylint: disable-next=unpacking-non-sequence
        x_utm, y_utm = transformer.transform(x_values, y_values)

        start_time = datetime.datetime.now()

        if method == "statistical":
            outlier_array = (
                outlier_filter.epipolar_statistical_outlier_filtering(
                    x_utm,
                    y_utm,
                    z_values,
                    half_window_size=kwargs["statistical.half_window_size"],
                    k=kwargs["statistical.k"],
                    dev_factor=kwargs["statistical.dev_factor"],
                    use_median=kwargs["statistical.use_median"],
                )
            )
        elif method == "small_components":
            outlier_array = (
                outlier_filter.epipolar_small_components_outlier_filtering(
                    x_utm,
                    y_utm,
                    z_values,
                    radius=kwargs["small_components.radius"],
                    min_cluster_size=kwargs[
                        "small_components.min_cluster_size"
                    ],
                    half_window_size=kwargs[
                        "small_components.half_window_size"
                    ],
                    clusters_distance_threshold=kwargs[
                        "small_components.clusters_distance_threshold"
                    ],
                )
            )

        end_time = datetime.datetime.now()
        filtering_duration = end_time - start_time
        print(f"Filtering duration: {filtering_duration}")

        profile_uint = x_ds.profile

        profile_uint.update(
            dtype=rasterio.uint16,
            count=1,
            nodata=None,
            transform=None,
            compress="lzw",
        )
        with rasterio.open(outlier_mask, "w", **profile_uint) as dst:
            dst.write(outlier_array.astype(rasterio.uint16), 1)

        profile_float = x_ds.profile

        profile_float.update(count=1, transform=None, compress="lzw")

        with rasterio.open(out_epipolar_x_image, "w", **profile_float) as dst:
            dst.write(x_utm, 1)
        with rasterio.open(out_epipolar_y_image, "w", **profile_float) as dst:
            dst.write(y_utm, 1)
        with rasterio.open(out_epipolar_z_image, "w", **profile_float) as dst:
            dst.write(z_values, 1)


def console_script():
    """Console script for filter_epipolar."""
    parser = argparse.ArgumentParser()

    # Common parameters
    parser.add_argument("in_epipolar_x_image")
    parser.add_argument("in_epipolar_y_image")
    parser.add_argument("in_epipolar_z_image")
    parser.add_argument("out_outliers")
    parser.add_argument("out_epipolar_x_image")
    parser.add_argument("out_epipolar_y_image")
    parser.add_argument("out_epipolar_z_image")
    parser.add_argument("epsg", type=int)

    subparsers = parser.add_subparsers(help="filtering method", dest="method")

    statistical_parser = subparsers.add_parser("statistical")
    small_components_parser = subparsers.add_parser("small_components")

    # Statistical filtering parameters
    statistical_parser.add_argument(
        "statistical.half_window_size", nargs="?", type=int, default=5
    )
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
        "small_components.half_window_size", nargs="?", type=int, default=5
    )
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
