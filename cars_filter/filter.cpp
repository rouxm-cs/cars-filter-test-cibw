// Copyright (C) 2024 Centre National d'Etudes Spatiales (CNES).
//
// This file is part of cars-filter
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "small_components_filtering.h"
#include "statistical_filtering.h"
#include "image.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "KDTree.h"



namespace py = pybind11;


template<typename T>
cars_filter::Image<T> pyarray_to_image(py::array_t<T> input_array)
{
  py::buffer_info info = input_array.request();
  return cars_filter::Image<T>(static_cast<T*>(info.ptr), info.shape[0], info.shape[1]);
}


py::array_t<double, py::array::c_style> pyEpipolarOutlierFiltering(
        py::array_t<double, py::array::c_style | py::array::forcecast>& x_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& y_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& z_values,
        const std::string& method
)
{
  // Build image wrappers around numpy arrays
  auto x_image = pyarray_to_image<double>(x_values);
  auto y_image = pyarray_to_image<double>(y_values);
  auto z_image = pyarray_to_image<double>(z_values);

  std::cout << "nrow ncol " << x_image.number_of_rows() << " " <<  x_image.number_of_cols() << std::endl;

  // // Create a Python object
  py::array_t<double, py::array::c_style> outlier_array({x_image.number_of_rows(),  x_image.number_of_cols()},
                                                        {x_image.number_of_cols() * sizeof(double),  sizeof(double)});
  auto outlier_image = pyarray_to_image<double>(outlier_array);

  if (method == "statistical_filtering")
  {
    std::cout << "Epipolar Outlier Filtering" << std::endl;
    epipolar_statistical_filtering(x_image, y_image, z_image, outlier_image);
  }
  else if (method == "small_components_filtering")
  {
    std::cout << "Small components filtering" << std::endl;
    epipolar_small_components_filtering(x_image, y_image, z_image, outlier_image);
  }

  return outlier_array;
}


py::array_t<double, py::array::c_style> pyEpipolarStatisticalOutlierFiltering(
        py::array_t<double, py::array::c_style | py::array::forcecast>& x_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& y_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& z_values,
        const unsigned int k,
        const unsigned int half_window_size,
        const double dev_factor,
        const double use_median
)
{
  // Build image wrappers around numpy arrays
  auto x_image = pyarray_to_image<double>(x_values);
  auto y_image = pyarray_to_image<double>(y_values);
  auto z_image = pyarray_to_image<double>(z_values);

  // // Create a Python object
  py::array_t<double, py::array::c_style> outlier_array({x_image.number_of_rows(),  x_image.number_of_cols()},
                                                        {x_image.number_of_cols() * sizeof(double),  sizeof(double)});
  auto outlier_image = pyarray_to_image<double>(outlier_array);

  epipolar_statistical_filtering(x_image, y_image, z_image, outlier_image, k, half_window_size, dev_factor, use_median);

  return outlier_array;
}



py::array_t<double, py::array::c_style> pyEpipolarSmallComponentsOutlierFiltering(
        py::array_t<double, py::array::c_style | py::array::forcecast>& x_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& y_values,
        py::array_t<double, py::array::c_style | py::array::forcecast>& z_values,
        const unsigned int min_cluster_size = 15,
        const double radius = 10,
        const unsigned int half_window_size = 5,
        const double clusters_distance_threshold = 4
)
{
  // Build image wrappers around numpy arrays
  auto x_image = pyarray_to_image<double>(x_values);
  auto y_image = pyarray_to_image<double>(y_values);
  auto z_image = pyarray_to_image<double>(z_values);

  std::cout << "nrow ncol " << x_image.number_of_rows() << " " <<  x_image.number_of_cols() << std::endl;

  // // Create a Python object
  py::array_t<double, py::array::c_style> outlier_array({x_image.number_of_rows(),  x_image.number_of_cols()},
                                                        {x_image.number_of_cols() * sizeof(double),  sizeof(double)});
  auto outlier_image = pyarray_to_image<double>(outlier_array);

  epipolar_small_components_filtering(x_image, y_image, z_image, outlier_image, min_cluster_size, radius, half_window_size, clusters_distance_threshold);

  return outlier_array;
}




py::list pyPointCloudStatisticalOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> x_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> y_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> z_array,
          const double dev_factor,
          const unsigned int k,
          const bool use_median)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info x_info = x_array.request();
  auto x_coords = static_cast<double *>(x_info.ptr);

  py::buffer_info y_info = y_array.request();
  auto y_coords = static_cast<double *>(y_info.ptr);

  py::buffer_info z_info = z_array.request();
  auto z_coords = static_cast<double *>(z_info.ptr);

  auto result = cars_filter::statistical_filtering(x_coords, y_coords, z_coords, x_info.shape[0], dev_factor, k, use_median);

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}


py::list pyPointCloudSmallComponentsOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> x_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> y_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> z_array,
          const double radius = 3,
          const unsigned int min_cluster_size = 15,
          const double clusters_distance_threshold = 4)
{
  std::cout << radius << " " << min_cluster_size << std::endl;
  /* Request a buffer descriptor from Python */
  py::buffer_info x_info = x_array.request();
  auto x_coords = static_cast<double *>(x_info.ptr);

  py::buffer_info y_info = y_array.request();
  auto y_coords = static_cast<double *>(y_info.ptr);

  py::buffer_info z_info = z_array.request();
  auto z_coords = static_cast<double *>(z_info.ptr);

  auto result = cars_filter::point_cloud_small_components_filtering(
                                        x_coords,
                                        y_coords,
                                        z_coords,
                                        x_info.shape[0],
                                        radius,
                                        min_cluster_size,
                                        clusters_distance_threshold
                                        );

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}




// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_small_components_outlier_filtering",
        &pyPointCloudSmallComponentsOutlierFiltering,
        "Filter outliers from point cloud using statistical method"
        );

  m.def("pc_statistical_outlier_filtering",
    &pyPointCloudStatisticalOutlierFiltering,
    "Filter outliers from point cloud using small components method"
    );

  m.def("epipolar_small_components_outlier_filtering",
        &pyEpipolarSmallComponentsOutlierFiltering,
        "Filter outliers from depth map in epipolar geometry",
        py::return_value_policy::take_ownership
        );

  m.def("epipolar_statistical_outlier_filtering",
        &pyEpipolarStatisticalOutlierFiltering,
        "Filter outliers from depth map in epipolar geometry",
        py::return_value_policy::take_ownership
        );
}