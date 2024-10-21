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

namespace cars_filter
{

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

  std::vector<unsigned int> m_indices;

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


inline bool operator> (const NeighborNode& lhs, const NeighborNode& rhs) 
{
  return lhs.distance > rhs.distance;
}

template <class Container>
class Adapter : public Container {
public:
    typedef typename Container::container_type container_type;
    container_type &get_container() { return this->c; }
};


class KDTree
{
public:
  // typedef for a structure containing a node and its distance to a given point
  //using NeighborNode = std::pair<KDNode*, double>;

  using KNeighborListBase = std::priority_queue<NeighborNode>;
  using KNeighborList = Adapter<KNeighborListBase>;

  KDTree(const PointCloud& point_cloud, unsigned int leaf_size = 1): m_point_cloud(point_cloud), m_leaf_size(leaf_size)
  {
    m_nodes.reserve(point_cloud.size());
    m_root_node = build_tree();

    m_x = m_point_cloud.m_coords[0];
    m_y = m_point_cloud.m_coords[1];
    m_z = m_point_cloud.m_coords[2];
  }

  inline double squared_euclidian_distance(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) const
  {
    const double dx = x1-x2;
    const double dy = y1-y2;
    const double dz = z1-z2;
    return dx * dx + dy * dy + dz * dz; 
  }

  inline double squared_euclidian_distance(double x, double y, double z, KDNode& node) const
  {
    return squared_euclidian_distance(x,
                              y,
                              z,
                              m_point_cloud.m_coords[0][node.m_idx],
                              m_point_cloud.m_coords[1][node.m_idx],
                              m_point_cloud.m_coords[2][node.m_idx]);
  }

  inline double squared_euclidian_distance(const PointType& point1, const PointType& point2) const
  {
    return squared_euclidian_distance(point1[0],
                              point1[1],
                              point1[2],
                              point2[0],
                              point2[1],
                              point2[2]);
  }

  inline double squared_euclidian_distance(const PointType& point, const KDNode& node) const
  {
    const double dx = point[0] - m_point_cloud.m_x[node.m_idx];
    const double dy = point[1] - m_point_cloud.m_y[node.m_idx];
    const double dz = point[2] - m_point_cloud.m_z[node.m_idx];

    return dx * dx + dy * dy + dz * dz;
  }

  inline double squared_euclidian_distance(const PointType& point, const unsigned int idx) const
  {
    const double dx = point[0] - m_x[idx];
    const double dy = point[1] - m_y[idx];
    const double dz = point[2] - m_z[idx];

    return dx * dx + dy * dy + dz * dz;
  }

  std::vector<KDNode>& getNodes()
  {
    return m_nodes;
  }


  void processLeaf(const PointType& point, KDNode* node, double& best_distance, KNeighborList& k_nearest_neighbors, const unsigned int k=1);

  void processLeafBall(const PointType& point, KDNode* node, std::vector<unsigned int>& neighbors, double squared_radius);

  std::vector<unsigned int> epipolar_neighbors_in_ball(double x, double y, double z, double radius);

  std::vector<NeighborNode> findKNNIterative(const PointType& point, unsigned int k=1, KDNode* starting_node= nullptr);


private:

  KDNode* build_tree();

  template <typename iterator_type>
  KDNode* grow_tree(iterator_type start_idx_it,
                    unsigned int number_of_points,
                    const unsigned int current_dimension,
                    KDNode* parent_node)
  {
    // We arrived at a leaf !
    if (number_of_points==0)
    {
      return nullptr;
    }

    if (number_of_points <= m_leaf_size)
    {
      //std::cout << "creating leaf node" << std::endl;
      KDNode node(*start_idx_it, current_dimension);

      node.m_indices.reserve(number_of_points);
      for (auto it = start_idx_it; it<start_idx_it+number_of_points; it++ )
      {
        node.m_indices.push_back(*it);
      }

      node.m_parent = parent_node;

      m_nodes.push_back(node);
      return &m_nodes.back();
    }

    auto median_pos= static_cast<unsigned int>(number_of_points/2);

    std::nth_element(start_idx_it,
                     start_idx_it+median_pos,
                     start_idx_it+number_of_points,
                     [this, current_dimension](int lhs, int rhs){return m_point_cloud.m_coords[current_dimension][lhs]<m_point_cloud.m_coords[current_dimension][rhs];});

    KDNode node(*(start_idx_it+median_pos), current_dimension);

    unsigned int next_dimension = (current_dimension +1) % 3;

    m_nodes.push_back(node);
    KDNode* node_ptr = &m_nodes.back();

    node_ptr->m_parent = parent_node;
    node_ptr->m_left_child = grow_tree(start_idx_it, median_pos, next_dimension, node_ptr);
    node_ptr->m_right_child = grow_tree(start_idx_it+median_pos+1, number_of_points-median_pos-1, next_dimension, node_ptr);

    return node_ptr;
  }


  PointCloud m_point_cloud;
  std::vector<KDNode> m_nodes;
  KDNode* m_root_node;

  double* m_x;
  double* m_y;
  double* m_z;

  unsigned int m_leaf_size;
};

} //namespace cars_filter
