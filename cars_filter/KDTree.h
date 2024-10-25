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

#include <queue>
#include <stack>

#include "utils.h"

namespace cars_filter
{

// Point cloud abstraction (does not own any data)
class PointCloud
{
public:
  // Number of points in the point cloud
  unsigned int m_num_elem;

  // x,y,z arrays as an array of array, used to fetch data of an integer dimension 
  std::array<double*, 3> m_coords;

  // number of element getter
  unsigned int size() const
  {
    return m_num_elem;
  }

  // Constructor from buffers
  PointCloud(double* x_coords, double* y_coords, double* z_coords, unsigned int num_elem)
    : m_num_elem(num_elem), m_coords({x_coords, y_coords, z_coords})
  {
  }

};


// A node in the KDTree
class KDNode
{
public:
  // Index of the node
  unsigned int m_idx;

  // Cut dimension of the node
  unsigned int m_dimension;

  // Leaf node special case: store a list of indices for brute force computation
  std::vector<unsigned int> m_indices;

  // Left children node in the tree
  KDNode* m_left_child;
  // Right children node in the tree
  KDNode* m_right_child;
  // Parent node
  KDNode* m_parent;

  KDNode(unsigned int idx, unsigned int dimension): m_idx(idx),
                                                    m_dimension(dimension),
                                                    m_left_child(nullptr),
                                                    m_right_child(nullptr), 
                                                    m_parent(nullptr)
  {
  };
};

// Structure containing a node and a distance used internally in KNN computations
struct NeighborNode
{
  KDNode* node;
  double distance;

  NeighborNode(KDNode* in_node, double in_distance): 
          node(in_node), distance(in_distance)
  {
  }

};

// Define operator< and operator> for std::greater and std::less
inline bool operator< (const NeighborNode& lhs, const NeighborNode& rhs) 
{
  return lhs.distance < rhs.distance;
}


inline bool operator> (const NeighborNode& lhs, const NeighborNode& rhs) 
{
  return lhs.distance > rhs.distance;
}

// Trick to give access to the internal vector of the priority queue, without
// having to pop the elements one by one
template <class Container>
class Adapter : public Container {
public:
    typedef typename Container::container_type container_type;
    container_type &get_container() { return this->c; }
};

/*
* \brief 3D KDTree implementation
*
* \details Implements a KDTree structure to perform spatial queries in a 3D 
* point cloud. The tree is constructed from a point cloud and provide query
* methods (KNN and radius based). Many implementations details are inspired
* by Scipy CKDtree implementation 
* (https://github.com/scipy/scipy/tree/main/scipy/spatial/ckdtree/src).
* In particular, a leaf size can be provided to the constructor of the tree,
* corresponding to the number of points stored on the leaf. When a leaf node is
* reached, all point distances at the node are computed. This brute force
* approach increases performances.
*/
class KDTree
{
public:
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

  // Euclidian distance functions between point and tree node
  inline double squared_euclidian_distance_in_tree(const PointType& point, const KDNode& node) const
  {
    const double dx = point[0] - m_x[node.m_idx];
    const double dy = point[1] - m_y[node.m_idx];
    const double dz = point[2] - m_z[node.m_idx];

    return dx * dx + dy * dy + dz * dz;
  }

  inline double squared_euclidian_distance_in_tree(double x, double y, double z, const unsigned int idx) const
  {
    const double dx = x - m_x[idx];
    const double dy = y - m_y[idx];
    const double dz = z - m_z[idx];

    return dx * dx + dy * dy + dz * dz;
  }

  // getter on the internal nodes
  std::vector<KDNode>& getNodes()
  {
    return m_nodes;
  }

  // Find all neighbors of point (x,y,z) in given radius
  std::vector<unsigned int> neighbors_in_ball(double x, double y, double z, double radius);

  // Find k nearest neighbor nodes of point (x,y,z)
  std::vector<NeighborNode> findKNNIterative(double x, double y, double z, unsigned int k=1, KDNode* starting_node= nullptr);

private:

  // Brute force KNN distance computation on a multi-point leaf node
  void processLeaf(double x, double y, double z, KDNode* node, double& best_distance, KNeighborList& k_nearest_neighbors, const unsigned int k=1);

  // Brute force radius distance computation on a multi-point leaf node
  void processLeafBall(const PointType& point, KDNode* node, std::vector<unsigned int>& neighbors, double squared_radius);

  // Initial function building the KDTree, calls grow_tree recursively and return root node of the tree
  KDNode* build_tree();

  // Recursive function to add node to a KDTree, until the whole point cloud has been added.
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

    // Median computation
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

inline double compute_axis_distance(const unsigned int dimension, const double x, const double y, const double z, const unsigned int idx)
{
  if (dimension == 0)
  {
    return x-m_x[idx];
  }
  else if (dimension == 1)
  {
    return y-m_y[idx];
  }
  else
  {
    return z-m_z[idx];
  }

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
