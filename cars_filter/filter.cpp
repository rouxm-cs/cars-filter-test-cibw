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

  // TODO: remove debug info
  std::cout << "info.itemsize " << info.itemsize << std::endl;
  std::cout << "info.format " << info.format << std::endl;
  std::cout << "info.ndim " << info.ndim << std::endl;

  for (const auto& elem: info.shape)
  {
    std::cout << "shape " << elem << std::endl;
  }
  for (const auto& elem: info.strides)
  {
    std::cout << "strides " << elem << std::endl;
  }
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

  // // Create a Python object
  py::array_t<double, py::array::c_style> outlier_array({x_image.number_of_rows(),  x_image.number_of_cols()},
                                                        {6520,  8});
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


py::list pyPointCloudStatisticalOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> x_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> y_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> z_array,
          const double dev_factor,
          const unsigned int k)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info x_info = x_array.request();
  auto x_coords = static_cast<double *>(x_info.ptr);

  py::buffer_info y_info = y_array.request();
  auto y_coords = static_cast<double *>(y_info.ptr);

  py::buffer_info z_info = z_array.request();
  auto z_coords = static_cast<double *>(z_info.ptr);

  auto result = cars_filter::statistical_filtering(x_coords, y_coords, z_coords, x_info.shape[0], dev_factor, k);

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
          const unsigned int min_cluster_size = 15)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info x_info = x_array.request();
  auto x_coords = static_cast<double *>(x_info.ptr);

  py::buffer_info y_info = y_array.request();
  auto y_coords = static_cast<double *>(y_info.ptr);

  py::buffer_info z_info = z_array.request();
  auto z_coords = static_cast<double *>(z_info.ptr);

  auto result = cars_filter::point_cloud_small_components_filtering_v2(x_coords, y_coords, z_coords, x_info.shape[0], radius, min_cluster_size);

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}


py::list pyOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> x_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> y_array,
          py::array_t<double,
          py::array::c_style | py::array::forcecast> z_array,
          const std::string& method)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info x_info = x_array.request();
  auto x_coords = static_cast<double *>(x_info.ptr);

  py::buffer_info y_info = y_array.request();
  auto y_coords = static_cast<double *>(y_info.ptr);

  py::buffer_info z_info = z_array.request();
  auto z_coords = static_cast<double *>(z_info.ptr);

  // TODO: remove debug info
  std::cout << "info.itemsize " << x_info.itemsize << std::endl;
  std::cout << "info.format " << x_info.format << std::endl;
  std::cout << "info.ndim " << x_info.ndim << std::endl;

  for (const auto& elem: x_info.shape)
  {
    std::cout << "shape " << elem << std::endl;
  }
  for (const auto& elem: x_info.strides)
  {
    std::cout << "strides " << elem << std::endl;
  }

  std::vector<unsigned int> result;
  if (method == "statistical_filtering")
  {
    result = cars_filter::statistical_filtering(x_coords, y_coords, z_coords, x_info.shape[0]);
  }
  else if (method == "small_components_filtering")
  {
    result = cars_filter::point_cloud_small_components_filtering_v2(x_coords, y_coords, z_coords, x_info.shape[0]);
  }

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}



// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_outlier_filtering", &pyOutlierFiltering, "Filter outliers from point cloud");
  m.def("pc_small_components_outlier_filtering", &pyPointCloudSmallComponentsOutlierFiltering, "Filter outliers from point cloud using statistical method");
  m.def("pc_statistical_outlier_filtering", &pyPointCloudStatisticalOutlierFiltering, "Filter outliers from point cloud using small components method");
  m.def("epipolar_outlier_filtering", &pyEpipolarOutlierFiltering, "Filter outliers from point cloud");
}