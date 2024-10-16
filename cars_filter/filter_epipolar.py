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
import sys

import pyproj
import rasterio

import outlier_filter


def main(epipolar_x_image, epipolar_y_image, epipolar_z_image):
    """
    Filter outliers in epipolar images
    """

    with rasterio.open(epipolar_x_image) as x_ds, rasterio.open(
        epipolar_y_image
    ) as y_ds, rasterio.open(epipolar_z_image) as z_ds:
        print("images opened!!")

        print(x_ds)
        print(y_ds)
        print(z_ds)

        x_values = x_ds.read(1)
        print(x_values.shape)
        y_values = y_ds.read(1)
        z_values = z_ds.read(1)

        print(f"x_values.stride {x_values.strides}")

        # hard code UTM 36N for Gizeh for now
        transformer = pyproj.Transformer.from_crs(4326, 32636)
        # X-Y inversion required because WGS84 is lat first ?
        x_utm, y_utm = transformer.transform(x_values, y_values)

        outlier_array = outlier_filter.epipolar_outlier_filtering(
            x_utm, y_utm, z_values, "statistical_filtering"
        )
        profile_uint = x_ds.profile

        print(outlier_array)

        print(profile_uint)

        profile_uint.update(
            dtype=rasterio.uint16,
            count=1,
            nodata=None,
            transform=None,
            compress="lzw",
        )
        with rasterio.open("example.tif", "w", **profile_uint) as dst:
            dst.write(outlier_array.astype(rasterio.uint16), 1)

        profile_float = x_ds.profile

        profile_float.update(count=1, transform=None, compress="lzw")
        print(x_values.shape)

        with rasterio.open("x_filtered.tif", "w", **profile_float) as dst:
            dst.write(x_utm, 1)
        with rasterio.open("y_filtered.tif", "w", **profile_float) as dst:
            dst.write(y_utm, 1)
        with rasterio.open("z_filtered.tif", "w", **profile_float) as dst:
            dst.write(z_values, 1)


def console_script():
    """Console script for filterlaz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("epipolar_x_image")
    parser.add_argument("epipolar_y_image")
    parser.add_argument("epipolar_z_image")
    args = parser.parse_args()
    main(args.epipolar_x_image, args.epipolar_y_image, args.epipolar_z_image)


if __name__ == "__main__":
    sys.exit(console_script())  # pragma: no cover
