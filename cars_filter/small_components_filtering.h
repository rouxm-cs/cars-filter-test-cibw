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


#include <unordered_set>

#include "KDTree.h"
#include "image.h"
#include "epipolar_utils.h"

namespace cars_filter
{


std::vector<unsigned int> point_cloud_small_components_filtering_v2(
                                                          double* x_coords,
                                                          double* y_coords,
                                                          double* z_coords,
                                                        const unsigned int num_elem,
                                                        const double radius = 3,
                                                        const int min_cluster_size = 15)
{
  std::cout << "num_elem" << num_elem << std::endl;

  auto input_point_cloud = PointCloud(x_coords, y_coords, z_coords, num_elem);

  auto tree = KDTree(input_point_cloud, 1);

  std::vector<unsigned int> visited_points(num_elem, 0);

  std::vector<unsigned int> result;

  // Iterate on the point cloud, and get the index of a point that have not been
  // processed yet
  for (unsigned int i=0; i< input_point_cloud.size(); i++)
  {
    auto idx = tree.getNodes()[i].m_idx;
    if (visited_points[idx])
    {
      continue;
    }

    // Get the neighbors of the point
    auto neighbor_list = tree.epipolar_neighbors_in_ball(input_point_cloud.m_x[idx], 
                   input_point_cloud.m_y[idx],
                   input_point_cloud.m_z[idx],
                   radius);

    // Add the neighbors to the current cluster
    std::unordered_set<unsigned int> seed;
    for (auto& elem: neighbor_list)
    {
      visited_points[elem] = true;
      seed.insert(elem);
    }

    while (!neighbor_list.empty())
    {

      auto current_idx = neighbor_list.back();
      neighbor_list.pop_back();

      // visited_points[current_idx] = true;
      auto new_neighbors = tree.epipolar_neighbors_in_ball(input_point_cloud.m_x[current_idx], 
                   input_point_cloud.m_y[current_idx],
                   input_point_cloud.m_z[current_idx],
                   radius);

      for (auto elem: new_neighbors)
      {
        auto [it, is_new_elem] = seed.insert(elem);
        if (is_new_elem)
        {
          visited_points[elem] = true;
          neighbor_list.push_back(elem);
        }
      }

    }

    std::cout << seed.size() << std::endl;

    if (seed.size() < min_cluster_size)
    {
      for (auto elem: seed)
      {
        result.push_back(elem);
      }
    }
  }
  return result;
}


std::vector<unsigned int> point_cloud_small_components_filtering(
                                                          double* x_coords,
                                                          double* y_coords,
                                                          double* z_coords,
                                                        unsigned int num_elem)
{

  auto input_point_cloud = PointCloud(x_coords, y_coords, z_coords, num_elem);

  auto tree = KDTree(input_point_cloud);

  double radius = 3;
  unsigned int min_cluster_size = 15;

  std::cout << "num_elem" << num_elem << std::endl;


  std::vector<unsigned int> visited_points(num_elem, 0);

  std::vector<unsigned int> clusters;

  std::vector<unsigned int> result;

  for (unsigned int i=0; i< input_point_cloud.size(); i++)
  {
    if (visited_points[tree.getNodes()[i].m_idx])
    {
      continue;
    }
    clusters.push_back(tree.getNodes()[i].m_idx);

    // if (visited_points[i])
    // {
    //   continue;
    // }
    // clusters.push_back(i);

    std::vector<unsigned int> current_cluster;
    visited_points[ clusters.back()] = true;
    while (!clusters.empty())
    {
      auto current_idx = clusters.back();

      clusters.pop_back();
      // if (visited_points[current_idx])
      // {
      //   continue;
      // }
      current_cluster.push_back(current_idx);
      // visited_points[current_idx] = true;
      auto neighbors = tree.epipolar_neighbors_in_ball(input_point_cloud.m_x[current_idx], 
                   input_point_cloud.m_y[current_idx],
                   input_point_cloud.m_z[current_idx],
                   radius);
      //clusters.reserve(clusters.size() + neighbors.size());
      for (auto idx : neighbors)
      {
        if (!visited_points[idx])
        {
          visited_points[idx] = true;
          clusters.push_back(idx);
        }
      }
    }

    if (current_cluster.size() < min_cluster_size)
    {
      for (auto elem: current_cluster)
      {
        result.push_back(elem);
      }
    }
  }

  std::cout << "result size" << result.size() << "/" << num_elem << std::endl;

  return result;
}


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
        //clusters.insert(clusters.end(), neighbors.begin(), neighbors.end());
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