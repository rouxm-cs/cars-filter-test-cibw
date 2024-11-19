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

#include "epipolar_utils.h"

#include <algorithm>

namespace cars_filter
{
double epipolar_knn(const Image<double>& x_coords,
                    const Image<double>& y_coords,
                    const Image<double>& z_coords,
                    unsigned int ref_row,
                    unsigned int ref_col,
                    unsigned int k,
                    unsigned int half_window_size)
{
  PointType ref_point = {x_coords.get(ref_row, ref_col),
                         y_coords.get(ref_row, ref_col),
                         z_coords.get(ref_row, ref_col)};

  // Don't look for the neighbors of an invalid point
  if (isnan(ref_point))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }

  unsigned int start_row = std::max(0, static_cast<int>(ref_row)-static_cast<int>(half_window_size));
  unsigned int end_row = std::min(x_coords.number_of_rows(), ref_row+half_window_size);
  unsigned int start_col = std::max(0, static_cast<int>(ref_col)-static_cast<int>(half_window_size));
  unsigned int end_col = std::min(x_coords.number_of_cols(), ref_col+half_window_size);

  std::vector<double> distances;

  for (unsigned int row = start_row; row<end_row; row++)
  {
    for (unsigned int col = start_col; col<end_col; col++)
    {
      PointType point = {x_coords.get(row, col),
                         y_coords.get(row, col),
                         z_coords.get(row, col)};

      if (!isnan(point))
      {
        auto dist = squared_euclidian_distance(point, ref_point);
        distances.push_back(squared_euclidian_distance(point, ref_point));
      }
    }
  }
  // No enough valid points in the epipolar neighbordhood, return NaN
  if (distances.size() < k)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Partial sort of the distances
  std::nth_element(distances.begin(), distances.begin() + k, distances.end());

  // apply sqrt on k first sorted distances
  std::transform(distances.begin(),
                 distances.begin() + k,
                 distances.begin(),
                 [](double val){return std::sqrt(val);});


  // return mean of k first sorted distances
  // Note that we divide by k-1 because it is assumed that the point itself
  // is in the neighborhood (distance = 0)
  return std::accumulate(distances.begin(), distances.begin() + k, 0.)/(k-1);
}

std::vector<std::pair<unsigned int, unsigned int>> epipolar_neighbors_in_ball(
                                              const Image<double>& x_coords,
                                              const Image<double>& y_coords,
                                              const Image<double>& z_coords,
                                              unsigned int ref_row,
                                              unsigned int ref_col,
                                              double radius,
                                              unsigned int half_window_size)
{
  const double squared_radius = radius*radius;

  std::vector<std::pair<unsigned int, unsigned int>> neighbors;

  PointType ref_point = {x_coords.get(ref_row, ref_col),
                         y_coords.get(ref_row, ref_col),
                         z_coords.get(ref_row, ref_col)};

  if (!isnan(ref_point))
  {
    unsigned int start_row = std::max(0, static_cast<int>(ref_row)-static_cast<int>(half_window_size));
    unsigned int end_row = std::min(x_coords.number_of_rows(), ref_row+half_window_size);
    unsigned int start_col = std::max(0, static_cast<int>(ref_col)-static_cast<int>(half_window_size));
    unsigned int end_col = std::min(x_coords.number_of_cols(), ref_col+half_window_size);

    for (unsigned int row = start_row; row<end_row; row++)
    {
      for (unsigned int col = start_col; col<end_col; col++)
      {
        PointType point = {x_coords.get(row, col),
                           y_coords.get(row, col),
                           z_coords.get(row, col)};
        if (squared_euclidian_distance(point, ref_point) <= squared_radius)
        {
          neighbors.push_back({row, col});
        }
      }
    }
  }

  return neighbors;
}

} // namespace cars_filter