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

#pragma once

#include <vector>
#include "KDTree.h"
#include "epipolar_utils.h"

namespace cars_filter
{

/*
*
* \brief Compute first quartile, median and third quartile of input vector
* 
* This function returns the closest element (rounded down) to the quantile.
* For example the median of a vector of 30 elements is the 15th element,
* not the mean between the 15th and 16th.
* 
* Note: the vector is copied because nth_element modifies the input container
*/
template<typename T>
std::tuple<T, T, T> compute_approximate_quantiles(std::vector<T> input_data)
{
  unsigned int size = input_data.size();
  unsigned int first_quarter_pos = static_cast<unsigned int>(size/4);
  unsigned int half_pos = static_cast<unsigned int>(size/2);
  unsigned int third_quarter_pos = first_quarter_pos + half_pos;

  // Split the container at the median
  std::nth_element(input_data.begin(),
                   input_data.begin() + half_pos,
                   input_data.end());

  // Split the first half of the container at the 0.25 quantile
  std::nth_element(input_data.begin(),
                   input_data.begin() + first_quarter_pos,
                   input_data.begin() + half_pos);

  // Split the other half of the container at the 0.75 quantile
  std::nth_element(input_data.begin() + half_pos+1,
                   input_data.begin() + third_quarter_pos,
                   input_data.end());

  return {input_data[first_quarter_pos], input_data[half_pos], input_data[third_quarter_pos]};
}


std::vector<unsigned int> statistical_filtering(double* x_coords,
                                                double* y_coords,
                                                double* z_coords,
                                                const unsigned int num_elem,
                                                const double dev_factor = 1.,
                                                const unsigned int k = 50,
                                                const bool use_median = false);

void epipolar_statistical_filtering(Image<double>& x_coords,
                                    Image<double>& y_coords,
                                    Image<double>& z_coords,
                                    Image<double>& outlier_array,
                                    const unsigned int k = 50,
                                    const unsigned int half_window_size = 15,
                                    const double dev_factor = 1,
                                    const double use_median = false);

} // namespace cars_filter