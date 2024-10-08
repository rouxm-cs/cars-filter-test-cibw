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

#include <array>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stack>

// Debug headers
#include <chrono>
#include <iostream>
#include <iomanip>

// TODO remove debug: recursion counter
static int num_recur = 0;


using PointType = std::array<double, 3>;

class PointCloud
{
public:
  // TODO: duplicated with coords, left here for easier debugging
  double* m_x;
  double* m_y;
  double* m_z;

  unsigned int m_num_elem;

  std::array<double*, 3> m_coords;

  unsigned int size() const
  {
    return m_num_elem;
  }

  // Constructor from buffers
  PointCloud(double* x_coords, double* y_coords, double* z_coords, unsigned int num_elem)
    : m_x(x_coords), m_y(y_coords), m_z(z_coords), m_num_elem(num_elem), m_coords({m_x, m_y, m_z})
  {
  }

};



class KDNode
{
public:
  unsigned int m_idx;
  // TODO: is that required in NN, or can we deduce it automatically
  unsigned int m_dimension;
  KDNode* m_left_child;
  KDNode* m_right_child;
  KDNode* m_parent;

  KDNode(unsigned int idx, unsigned int dimension): m_idx(idx), m_dimension(dimension), m_left_child(nullptr), m_right_child(nullptr), m_parent(nullptr)
  {
  };
};

struct NeighborNode
{
  KDNode* node;
  double distance;

  NeighborNode(KDNode* in_node, double in_distance): 
          node(in_node), distance(in_distance)
  {
  }

};

inline bool operator< (const NeighborNode& lhs, const NeighborNode& rhs) 
{
  return lhs.distance < rhs.distance;
}



class KDTree
{
public:
  // typedef for a structure containing a node and its distance to a given point
  //using NeighborNode = std::pair<KDNode*, double>;



  using KNeighborList = std::priority_queue<NeighborNode>;

  KDTree(const PointCloud& point_cloud): m_point_cloud(point_cloud)
  {
    m_nodes.reserve(point_cloud.size());
    m_root_node = build_tree();
  }

  // TODO better parameter handling ? Maybe create a Point Type
  // problem is that we use array of coords (x*N, y*N, z*N) and individual points
  // TODO Squared euclidian distance would be more efficient (but we need the true distance in filtering)
  double euclidian_distance(double x1, double y1, double z1, double x2, double y2, double z2) const
  {
    return std::sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)); 
  }


  double euclidian_distance(double x, double y, double z, KDNode& node) const
  {
    return euclidian_distance(x,
                              y,
                              z,
                              m_point_cloud.m_x[node.m_idx],
                              m_point_cloud.m_y[node.m_idx],
                              m_point_cloud.m_z[node.m_idx]);
  }

  double euclidian_distance(const PointType& point1, const PointType& point2) const
  {
    return euclidian_distance(point1[0],
                              point1[1],
                              point1[2],
                              point2[0],
                              point2[1],
                              point2[2]);
  }

  double euclidian_distance(const PointType& point, const KDNode& node) const
  {
    return euclidian_distance(point[0],
                              point[1],
                              point[2],
                              m_point_cloud.m_x[node.m_idx],
                              m_point_cloud.m_y[node.m_idx],
                              m_point_cloud.m_z[node.m_idx]);
  }

  NeighborNode findNN(double x, double y, double z)
  {
    num_recur = 0;
    std::cout << "find NN" << std::endl;
    double distance = std::numeric_limits<double>::max();

    NeighborNode nearest_neighbor = {nullptr, distance};

    searchNN({x, y, z}, m_root_node, nearest_neighbor);

    auto best_idx = nearest_neighbor.node->m_idx;


    std::cout <<  std::setprecision (15) 
             <<  "best match: " << m_point_cloud.m_x[best_idx] << " " 
                                << m_point_cloud.m_y[best_idx] << " " 
                                << m_point_cloud.m_z[best_idx] << std::endl;
    std::cout << "number of recursion in findNN: " << num_recur << std::endl;

    return nearest_neighbor;
  }



  NeighborNode findNNIterative(double x, double y, double z)
  {
    int number_of_iterations = 0;
    PointType point = {x, y, z};
    double distance = std::numeric_limits<double>::max();
    NeighborNode nearest_neighbor = {nullptr, distance};

    std::stack< std::pair<KDNode*, double> , std::vector< std::pair<KDNode*, double> > > node_queue;
    node_queue.push({m_root_node, 0} );

    while(!node_queue.empty())
    {
      auto [current_node, axis_distance] = node_queue.top();
      node_queue.pop();

      if (axis_distance >= nearest_neighbor.distance)
      {
        std::cout << "no need to go on the other side" << std::endl;
        continue;
      }

      std::cout << "a better match might be on the other side !" << std::endl;

      while (current_node)
      {
        number_of_iterations++;
        auto current_idx = current_node->m_idx;
        auto current_dimension = current_node->m_dimension;

        // check if current node is a better match
        auto current_distance = euclidian_distance(point, *current_node);
        if (current_distance < nearest_neighbor.distance)
        {
          nearest_neighbor = {current_node, current_distance};
        }

        std::cout << "current node:" << current_idx << "[" << current_distance << "], best: " 
                                    << nearest_neighbor.node->m_idx << "[" << nearest_neighbor.distance << "]" << std::endl; 

        bool is_left = point[current_dimension] < m_point_cloud.m_coords[current_dimension][current_idx];
        KDNode* next_node = is_left ? current_node->m_left_child : current_node->m_right_child;

        double axis_distance = std::abs(point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx]);

        KDNode* other_node = is_left ? current_node->m_right_child : current_node->m_left_child;
        node_queue.push({other_node, axis_distance});

        current_node = next_node;

      }
    }

    std::cout << "number of iteration:" << number_of_iterations << std::endl;

    return nearest_neighbor;
  }



  std::vector<NeighborNode> findKNNIterative(double x, double y, double z, unsigned int k=1)
  {
    int number_of_iterations = 0;
    PointType point = {x, y, z};

    // Initialize the best matches container
    KNeighborList k_nearest_neighbors;
    for (unsigned int i=0; i<k; i++)
    {
      k_nearest_neighbors.emplace(nullptr, std::numeric_limits<double>::max());
    }
    double best_distance = std::numeric_limits<double>::max();

    std::stack< std::pair<KDNode*, double> > node_queue;
    node_queue.push({m_root_node, 0} );

    while(!node_queue.empty())
    {
      auto [current_node, axis_distance] = node_queue.top();
      node_queue.pop();

      if (axis_distance >= best_distance)
      {
        //std::cout << "no need to go on the other side" << std::endl;
        continue;
      }

      //std::cout << "a better match might be on the other side !" << std::endl;

      while (current_node)
      {

        number_of_iterations++;
        auto current_idx = current_node->m_idx;
        auto current_dimension = current_node->m_dimension;

        // check if current node is a better match
        auto current_distance = euclidian_distance(point, *current_node);

        if (current_distance < best_distance)
        {
          // remove the worst neighbor from the list
          k_nearest_neighbors.pop();
          // add the current node to the list
          k_nearest_neighbors.emplace(current_node, current_distance);
          best_distance = k_nearest_neighbors.top().distance;
        }

        //std::cout << "best distance " << best_distance << std::endl;
        // std::cout << "current node:" << current_idx << "[" << current_distance << "], best: " 
        //                             << k_nearest_neighbors.top().node->m_idx << "[" << k_nearest_neighbors.top().distance << "]" << std::endl; 

        bool is_left = point[current_dimension] < m_point_cloud.m_coords[current_dimension][current_idx];
        KDNode* next_node = is_left ? current_node->m_left_child : current_node->m_right_child;

        double axis_distance = std::abs(point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx]);

        KDNode* other_node = is_left ? current_node->m_right_child : current_node->m_left_child;
        node_queue.push({other_node, axis_distance});

        current_node = next_node;

      }
    }

    //std::cout << "number_of_iterations " << number_of_iterations << std::endl;

    // convert priority queue to vector
    std::vector<NeighborNode> output_neighbors;

    for (unsigned int i=0; i<k; i++)
    {
      output_neighbors.push_back(k_nearest_neighbors.top());
      k_nearest_neighbors.pop();
    }

    // for (const auto& elem: output_neighbors)
    // {
    //   auto idx = elem.node->m_idx;
    //   std::cout << "neighbor:" << idx  << " "
    //                             << m_point_cloud.m_x[idx] << " "
    //                             << m_point_cloud.m_y[idx] << " "
    //                             << m_point_cloud.m_z[idx] << " "
    //                             << elem.distance << std::endl;
    // }
    // std::cout << "number of recursion in findNN: " << num_recur << std::endl;
    return output_neighbors;
  }




  std::vector<NeighborNode> findKNN(double x, double y, double z, unsigned int k=1)
  {
    num_recur = 0;
    // Initialize the best matches container
    KNeighborList k_nearest_neighbors;
    for (unsigned int i=0; i<k; i++)
    {
      k_nearest_neighbors.emplace(nullptr, std::numeric_limits<double>::max());
    }
    double initial_distance = std::numeric_limits<double>::max();

    PointType point = {x, y, z};
    searchKNN(point, m_root_node, k_nearest_neighbors, initial_distance);

    // convert priority queue to vector
    std::vector<NeighborNode> output_neighbors;

    for (unsigned int i=0; i<k; i++)
    {
      output_neighbors.push_back(k_nearest_neighbors.top());
      k_nearest_neighbors.pop();
    }

    // for (const auto& elem: output_neighbors)
    // {
    //   auto idx = elem.node->m_idx;
    //   std::cout << "neighbor:" << idx  << " "
    //                             << m_point_cloud.m_x[idx] << " "
    //                             << m_point_cloud.m_y[idx] << " "
    //                             << m_point_cloud.m_z[idx] << " "
    //                             << elem.distance << std::endl;
    // }
    // std::cout << "number of recursion in findNN: " << num_recur << std::endl;
    return output_neighbors;
  }

private:

  KDNode* build_tree()
  {
    std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

    auto indexes = std::vector<unsigned int>(m_point_cloud.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    // First split to find the root node
    KDNode* root_node = grow_tree(indexes.begin(), indexes.size(), 0, nullptr);

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    // TODO Debug info
    std::cout << "Tree build time" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms" << std::endl;
    std::cout << "number of nodes " << m_nodes.size() << std::endl;

    // for (const auto& elem: m_nodes)
    // {
    //   std::cout << elem.m_idx << std::endl;
    // }

    return root_node;
  }

  template <typename iterator_type>
  KDNode* grow_tree(iterator_type start_idx_it,
                    unsigned int number_of_points,
                    const unsigned int current_dimension,
                    KDNode* root_node = nullptr)
  {
    // We arrived at a leaf !
    if (number_of_points==0)
    {
      return nullptr;
    }

    auto median_pos= static_cast<unsigned int>(number_of_points/2);

    // on x for debug
    // std::nth_element(start_idx_it,
    //                  start_idx_it+median_pos,
    //                  start_idx_it+number_of_points,
    //                  [this](int lhs, int rhs){return m_point_cloud.m_x[lhs]<m_point_cloud.m_x[rhs];});

    std::nth_element(start_idx_it,
                     start_idx_it+median_pos,
                     start_idx_it+number_of_points,
                     [this, current_dimension](int lhs, int rhs){return m_point_cloud.m_coords[current_dimension][lhs]<m_point_cloud.m_coords[current_dimension][rhs];});

    KDNode node(*(start_idx_it+median_pos), current_dimension);

    unsigned int next_dimension = (current_dimension +1) % 3;

    node.m_left_child = grow_tree(start_idx_it, median_pos, next_dimension, &node);
    node.m_right_child = grow_tree(start_idx_it+median_pos+1, number_of_points-median_pos-1, next_dimension, &node);

    node.m_parent = root_node;

    m_nodes.push_back(node);

    return &m_nodes.back();
  }

  //TODO reference here for PointType
  void searchNN(const PointType point, KDNode* node, NeighborNode& best_match)
  {
    num_recur++;
    if (!node)
    {
      std::cout << "We arrived at a leaf!" << std::endl;
      return;
    }

    auto current_idx = node->m_idx;
    auto current_dimension = node->m_dimension;

    std::cout << "current_idx" << current_idx << std::endl;

    bool is_left = point[current_dimension] < m_point_cloud.m_coords[current_dimension][current_idx];
    KDNode* next_node = is_left ? node->m_left_child : node->m_right_child;

    // Go to this branch leaf
    searchNN(point, next_node, best_match);

    // check if current node is a better match
    auto current_distance = euclidian_distance(point, *node);

    if (current_distance < best_match.distance)
    {
      best_match = {node, current_distance};
    }

    std::cout << "current distance " << current_distance << " " << best_match.distance << std::endl;

    // Check we need to search for point in the other side of this node
    double axis_distance = std::abs(point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx]);

    if (axis_distance < best_match.distance)
    {
      std::cout << "a better match might be on the other side !" << std::endl;
      KDNode* other_node = is_left ? node->m_right_child : node->m_left_child;
      searchNN(point, other_node, best_match);
    }
    else
    {
      std::cout << "no need to go on the other side" << std::endl;
    }
  }

  void searchKNN(const PointType& point, KDNode* node, KNeighborList& best_matches, double& best_distance)
  {
    //num_recur++;
    if (!node)
    {
      //std::cout << "We arrived at a leaf!" << std::endl;
      return;
    }
    
    auto current_idx = node->m_idx;
    auto current_dimension = node->m_dimension;

    //std::cout << "current_idx" << current_idx << std::endl;

    bool is_left = point[current_dimension] < m_point_cloud.m_coords[current_dimension][current_idx];
    KDNode* next_node = is_left ? node->m_left_child : node->m_right_child;

    // check if current node is a better match
    auto current_distance = euclidian_distance(point, *node);

    if (current_distance < best_distance)
    {
      // remove the worst neighbor from the list
      best_matches.pop();
      // add the current node to the list
      best_matches.emplace(node, current_distance);
      best_distance = best_matches.top().distance;
    }
    // Go to this branch leaf
    searchKNN(point, next_node, best_matches, best_distance);

    // std::cout << "current distance: " << current_distance << " ,k best distance:" << best_matches.top().distance << std::endl;
    // std::cout << "size: " << best_matches.size();

    // Check we need to search for point in the other side of this node
    double axis_distance = std::abs(point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx]);

    if (axis_distance < best_distance)
    {
      //std::cout << "a better match might be on the other side !" << std::endl;
      KDNode* other_node = is_left ? node->m_right_child : node->m_left_child;
      searchKNN(point, other_node, best_matches, best_distance);
    }
  }

  PointCloud m_point_cloud;
  std::vector<KDNode> m_nodes;
  KDNode* m_root_node;
};


