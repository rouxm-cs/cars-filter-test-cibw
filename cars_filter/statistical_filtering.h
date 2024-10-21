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

#include <chrono>

namespace cars_filter
{

std::vector<unsigned int> statistical_filtering(double* x_coords,
                                                double* y_coords,
                                                double* z_coords,
                                                const unsigned int num_elem,
                                                const double dev_factor = 1.,
                                                const unsigned int k = 50,
                                                const bool use_median = false)
{
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto input_point_cloud = PointCloud(x_coords, y_coords, z_coords, num_elem);

  auto tree = KDTree(input_point_cloud, 40);

  // std::vector<double> mean_distances;
  // mean_distances.reserve(num_elem);
  // for (unsigned int i=0; i< input_point_cloud.size(); i++)
  // {
  //   const PointType point = {input_point_cloud.m_x[i], input_point_cloud.m_y[i], input_point_cloud.m_z[i]};

  //   auto k_neighbors = tree.findKNNIterative(point, k);
  //   double acc =0;
  //   for (const auto& elem : k_neighbors)
  //   {
  //     acc += std::sqrt(elem.distance);
  //   }
  //   mean_distances.push_back(acc/k);
  // }


  std::vector<double> mean_distances(num_elem, 0);
  for (auto& node: tree.getNodes())
  {
    // Leaf Node case
    if (node.m_indices.empty())
    {
      //const PointType point = {input_point_cloud.m_x[node.m_idx], input_point_cloud.m_y[node.m_idx], input_point_cloud.m_z[node.m_idx]};

      auto k_neighbors = tree.findKNNIterative(input_point_cloud.m_x[node.m_idx],
                                               input_point_cloud.m_y[node.m_idx],
                                               input_point_cloud.m_z[node.m_idx],
                                               k,
                                               &node);
      double acc =0;
      for (const auto& elem : k_neighbors)
      {
        acc += std::sqrt(elem.distance);
      }
      mean_distances[node.m_idx] = acc/k;
    }
    else
    {
      for (auto idx: node.m_indices)
      {
        //const PointType point = {input_point_cloud.m_x[idx], input_point_cloud.m_y[idx], input_point_cloud.m_z[idx]};
        auto k_neighbors = tree.findKNNIterative(input_point_cloud.m_x[idx],
                                                 input_point_cloud.m_y[idx],
                                                 input_point_cloud.m_z[idx],
                                                 k,
                                                 &node);
        double acc =0;
        for (const auto& elem : k_neighbors)
        {
          acc += std::sqrt(elem.distance);
        }
        mean_distances[idx] = acc/k;
      }
    }

  }


  double dist_thresh;

  if (use_median)
  {
    // TODO: median mode
  }
  else
  {
    // mean and variance mode
    const double mean = std::accumulate(mean_distances.begin(), mean_distances.end(), 0.0) / num_elem;

    // variance (mean)
    auto variance_lambda= [mean](double acc, double input) {
        return acc + (input - mean)*(input - mean);
    };

    const double var = std::accumulate(mean_distances.begin(), mean_distances.end(), 0.0, variance_lambda)/num_elem;
    const double stddev = std::sqrt(var);

    std::cout << "computed statistics " << mean << " " << var << " " << stddev << std::endl;

    dist_thresh = mean + dev_factor * stddev;
  }
  std::cout << "dist_thresh " << dist_thresh << std::endl;

  std::vector<unsigned int> result;

  // list version
  for (unsigned int i = 0; i<num_elem; i++)
  {
    if (mean_distances[i] > dist_thresh)
    {
      result.push_back(i);
    }
  }

  // mask version
  // std::transform(mean_distances.begin(), mean_distances.end(), result.begin(), 
  //       [dist_thresh](double input) {return input < dist_thresh;});

  // std::cout << "resultsize" << result.size() << std::endl;
  // for (unsigned int i=0; i<num_elem; i++)
  // {
  //   std::cout << static_cast<int>(elem) << " ";
  // }
  // std::cout << std::endl;

std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
std::cout << "statistical_filtering duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms." << std::endl;
  return result;
}


void epipolar_statistical_filtering(Image<double>& x_coords,
                                    Image<double>& y_coords,
                                    Image<double>& z_coords,
                                    Image<double>& outlier_array)
{
  unsigned k = 15;
  unsigned half_window_size = 7;

  InMemoryImage<double> mean_distance_image(x_coords.number_of_rows(), x_coords.number_of_cols());

  for (unsigned int row=0; row < x_coords.number_of_rows(); row++)
  {
    for (unsigned int col=0; col < x_coords.number_of_cols(); col++)
    {
      //std::cout << row << " " << col << std::endl;
      auto mean_dist = epipolar_knn(x_coords, y_coords, z_coords, row, col, k, half_window_size);
      mean_distance_image.get(row, col) = mean_dist;
    }
  }

  // mean with number of valid pixel counting
  unsigned int num_valid = 0;
  auto mean_lambda = [&num_valid](double acc, double input)
  {
    if (std::isnan(input))
    {
      return acc;
    }
    else
    {
      num_valid++;
      return acc + input;
    }
  };


  auto mean = std::accumulate(mean_distance_image.getData(), mean_distance_image.getData()+mean_distance_image.size(), 0.,mean_lambda );
  mean/=num_valid;

  std::cout << "mean " << mean << std::endl;


  // variance
  auto variance_lambda = [mean](double acc, double input)
  {
    if (std::isnan(input))
    {
      return acc;
    }
    else
    {
      return acc + (input - mean)*(input - mean);
    }
  };

  const double var = std::accumulate(mean_distance_image.getData(), mean_distance_image.getData()+mean_distance_image.size(), 0.0, variance_lambda)/num_valid;
  const double stddev = std::sqrt(var);

  std::cout << "var " << var << std::endl;

  const double dev_factor = 1;

  const double distance_threshold = mean + dev_factor * stddev;

  for (unsigned int row=0; row < x_coords.number_of_rows(); row++)
  {
    for (unsigned int col=0; col < x_coords.number_of_cols(); col++)
    {
      if (mean_distance_image.get(row, col) > distance_threshold)
      {
        outlier_array.get(row, col) = true;
        x_coords.get(row, col) = std::numeric_limits<double>::quiet_NaN();
        z_coords.get(row, col) = std::numeric_limits<double>::quiet_NaN();
        z_coords.get(row, col) = std::numeric_limits<double>::quiet_NaN();

      }
    }
  }



}

}