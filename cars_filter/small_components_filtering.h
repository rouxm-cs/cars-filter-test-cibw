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

#include "image.h"
#include "epipolar_utils.h"

namespace cars_filter
{


void epipolar_small_components_filtering(Image<double>& x_coords,
                                         Image<double>& y_coords,
                                         Image<double>& z_coords,
                                         Image<double>& outlier_array)
{

  std::cout << "In epipolar_small_components_filtering" << std::endl;

  unsigned int min_cluster_size = 15;

  double radius = 10;
  unsigned int half_window_size = 5;

  InMemoryImage<double> visited_pixels(x_coords.number_of_rows(), x_coords.number_of_cols());

  std::vector<std::pair<unsigned int, unsigned int>> clusters;

  for (unsigned int row=0; row < x_coords.number_of_rows(); row++)
  {
    for (unsigned int col=0; col < x_coords.number_of_cols(); col++)
    {
      if (std::isnan(z_coords.get(row, col)))
      {
        continue;
      }

      clusters.push_back({row,col});

      std::vector<std::pair<unsigned int, unsigned int>> current_cluster;
      while (!clusters.empty())
      {
        auto [ref_row, ref_col] = clusters.back();
        clusters.pop_back();
        auto& is_ref_visted = visited_pixels.get(ref_row, ref_col);
        if (is_ref_visted)
        {
          continue;
        }
        current_cluster.push_back({ref_row, ref_col});
        is_ref_visted = true;
        auto neighbors = (epipolar_neighbors_in_ball(x_coords, y_coords, z_coords, ref_row, ref_col, radius, half_window_size));

        //clusters.reserve(clusters.size() + neighbors.size());
        for (const auto& [elem_row, elem_col] :neighbors)
        {
          if (!visited_pixels.get(elem_row, elem_col))
          {
            clusters.push_back({elem_row, elem_col});
          }
        }
        clusters.insert(clusters.end(), neighbors.begin(), neighbors.end());
      }

      if (!current_cluster.empty())
      {
        if (current_cluster.size() < min_cluster_size)
        {
          std::cout << "removed current_cluster size" << current_cluster.size() << std::endl;
          for (const auto& [elem_row, elem_col]:current_cluster)
          {
            outlier_array.get(elem_row, elem_col) = current_cluster.size();
          }
        }
        else
        {
          std::cout << "current_cluster size" << current_cluster.size() << std::endl;
          for (const auto& [elem_row, elem_col]:current_cluster)
          {
            outlier_array.get(elem_row, elem_col) = current_cluster.size();
          }
        }
      }
    }
  }


}



}