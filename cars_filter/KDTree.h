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

  std::vector<unsigned int> m_indices;

  KDNode* m_left_child;
  KDNode* m_right_child;

  KDNode(unsigned int idx, unsigned int dimension): m_idx(idx), m_dimension(dimension), m_left_child(nullptr), m_right_child(nullptr)
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

  const std::vector<KDNode>& getNodes() const
  {
    return m_nodes;
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

    std::stack< std::pair<KDNode*, double> > node_queue;
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
        auto current_distance = squared_euclidian_distance(point, *current_node);
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



  inline void processLeaf(const PointType& point, KDNode* node, double& best_distance, KNeighborList& k_nearest_neighbors, const unsigned int k=1)
  {
    for (auto idx_it = node->m_indices.begin(); idx_it !=  node->m_indices.end(); idx_it++)
    {
      // check if current node is a better match
      auto current_distance = squared_euclidian_distance(point, *idx_it);
      //std::cout << current_distance << " ";
      if (current_distance < best_distance)
      {
        // remove the worst neighbor from the list
        if (k_nearest_neighbors.size() == k)
        {
          k_nearest_neighbors.pop();
        }

        // add the current node to the list
        k_nearest_neighbors.emplace(node, current_distance);
        if (k_nearest_neighbors.size() == k)
        {
          best_distance = k_nearest_neighbors.top().distance;
        }

      }
    }
  }

  void processLeafBall(const PointType& point, KDNode* node, std::vector<unsigned int>& neighbors, double squared_radius)
  {
    for (auto idx_it = node->m_indices.begin(); idx_it !=  node->m_indices.end(); idx_it++)
    {
      PointType node_point = {m_point_cloud.m_x[*idx_it], 
                              m_point_cloud.m_y[*idx_it], 
                              m_point_cloud.m_z[*idx_it]};
      // check if current node is in the ball around the point
      if (squared_euclidian_distance(point, node_point) < squared_radius)
      {
        //std::cout << squared_euclidian_distance(point, node_point) << " " << *idx_it << " " << squared_radius << std::endl;
        neighbors.push_back(*idx_it);
      }

    }
  }

  std::vector<unsigned int> epipolar_neighbors_in_ball(double x, double y, double z, double radius)
  {
    PointType point = {x, y, z};
    const double squared_radius = radius * radius;

    std::vector<unsigned int> neighbors;


    std::stack<KDNode*> node_queue;
    node_queue.push(m_root_node );

    while(!node_queue.empty())
    {
      auto current_node = node_queue.top();
      node_queue.pop();


      while (current_node)
      {
        if (!current_node->m_indices.empty())
        {
          processLeafBall(point, current_node, neighbors, squared_radius);
          break;
        }

        auto current_idx = current_node->m_idx;
        auto current_dimension = current_node->m_dimension;

        bool is_left = point[current_dimension] < m_point_cloud.m_coords[current_dimension][current_idx];
        KDNode* next_node = is_left ? current_node->m_left_child : current_node->m_right_child;

        if (squared_euclidian_distance(point, *current_node) < squared_radius)
        {
          //std::cout << squared_euclidian_distance(point, *current_node) << " " << current_node->m_idx << " " << squared_radius << std::endl;
          neighbors.push_back(current_node->m_idx);
        }

        double axis_distance = std::abs(point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx]);

        if(axis_distance < radius)
        {
          KDNode* other_node = is_left ? current_node->m_right_child : current_node->m_left_child;
          node_queue.push(other_node);
        }
        current_node = next_node;
      }
    }
    return neighbors;
  }


  std::vector<NeighborNode> findKNNIterative(const PointType& point, unsigned int k=1)
  {
    // Initialize the best matches container. Note that is empty at the 
    // beginning of the algorithm and is filled with neighbors until reaching k
    // elements
    KNeighborList k_nearest_neighbors;
    k_nearest_neighbors.get_container().reserve(k);

    double best_distance = std::numeric_limits<double>::max();

    Adapter<std::priority_queue< NeighborNode , std::vector<NeighborNode>, std::greater<NeighborNode> >> node_queue;
    node_queue.get_container().reserve(100);
    node_queue.emplace(m_root_node, 0);

    while(!node_queue.empty())
    {
      const auto current_neighbor = node_queue.top();
      auto current_node = current_neighbor.node;
      const auto current_axis_distance = current_neighbor.distance;
      node_queue.pop();

      // Do we have to look at this side of the tree ?
      if (current_axis_distance >= best_distance)
      {
        continue;
      }

      while (current_node)
      {
        // We arrived at a leaf, if it contains a single point, process as usual
        // If it contains several point, we use brute force to compute all 
        // distances in the leaf. This optimization helps reducing the number of
        // branchs in the tree, and is adapted from scipy ckdtree implementation.
        if (!current_node->m_indices.empty())
        {
          processLeaf(point, current_node, best_distance, k_nearest_neighbors, k);
          break;
        }

        const auto current_idx = current_node->m_idx;
        const auto current_dimension = current_node->m_dimension;

        // check if current node is a better match
        auto current_distance = squared_euclidian_distance(point, current_node->m_idx);

        if (current_distance < best_distance)
        {
          // remove the worst neighbor from the list
          if (k_nearest_neighbors.size() == k)
          {
            k_nearest_neighbors.pop();
          }
          // add the current node to the list
          k_nearest_neighbors.emplace(current_node, current_distance);
          if (k_nearest_neighbors.size() == k)
          {
            best_distance = k_nearest_neighbors.top().distance;
          }
        }

        // Find on which side of the tree the point of interest is
        const double axis_distance = point[current_dimension] - m_point_cloud.m_coords[current_dimension][current_idx];
        const bool is_left = axis_distance < 0;
        const double squared_axis_distance = axis_distance*axis_distance;
        KDNode* other_branch_node = is_left ? current_node->m_right_child : current_node->m_left_child;

        // Add the node on the other side of tree to the stack of node to be 
        // processed. The check verifying that neighbors can actually be on
        // this side is also done at the beginning of the loop, to avoid false 
        // negatives in case a smaller distance is found in this branch
        if (squared_axis_distance < best_distance)
        {
          node_queue.emplace(other_branch_node, squared_axis_distance);
        }
        // Go to next node on this branch
        current_node = is_left ? current_node->m_left_child : current_node->m_right_child;
      }
    }
    return k_nearest_neighbors.get_container();
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
    return output_neighbors;
  }

private:

  KDNode* build_tree()
  {
    std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

    auto indexes = std::vector<unsigned int>(m_point_cloud.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    // First split to find the root node
    KDNode* root_node = grow_tree(indexes.begin(), indexes.size(), 0);

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    // TODO Debug info
    std::cout << "Tree build time" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "ms" << std::endl;
    std::cout << "number of nodes " << m_nodes.size() << std::endl;

    return root_node;
  }

  template <typename iterator_type>
  KDNode* grow_tree(iterator_type start_idx_it,
                    unsigned int number_of_points,
                    const unsigned int current_dimension)
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

      m_nodes.push_back(node);
      return &m_nodes.back();
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

    node.m_left_child = grow_tree(start_idx_it, median_pos, next_dimension);
    node.m_right_child = grow_tree(start_idx_it+median_pos+1, number_of_points-median_pos-1, next_dimension);

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
    auto current_distance = squared_euclidian_distance(point, *node);

    if (current_distance < best_match.distance)
    {
      best_match = {node, current_distance};
    }

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
    auto current_distance = squared_euclidian_distance(point, *node);

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

  double* m_x;
  double* m_y;
  double* m_z;

  unsigned int m_leaf_size;
};


