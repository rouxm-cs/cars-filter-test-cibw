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

#include "statistical_filtering.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "KDTree.h"

namespace py = pybind11;


py::list pyOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> points)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info info = points.request();
  auto x_coords = static_cast<double *>(info.ptr);

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

  auto y_coords = x_coords + info.shape[1];
  auto z_coords = y_coords + info.shape[1];
  auto result = statistical_filtering(x_coords, y_coords, z_coords, info.shape[1]);

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}


// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_outlier_filtering", &pyOutlierFiltering, "Filter outliers from point cloud");
}