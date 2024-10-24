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

#include "KDTree.h"

#include "utils.h"

namespace cars_filter
{

std::vector<unsigned int> KDTree::neighbors_in_ball(double x,
                                                    double y,
                                                    double z,
                                                    double radius)
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

      if (squared_euclidian_distance_in_tree(point, *current_node) < squared_radius)
      {
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

std::vector<NeighborNode> KDTree::findKNNIterative(double x,
                                                   double y,
                                                   double z,
                                                   unsigned int k,
                                                   KDNode* starting_node)
{
  // Initialize the best matches container. Note that is empty at the 
  // beginning of the algorithm and is filled with neighbors until reaching k
  // elements
  KNeighborList k_nearest_neighbors;
  k_nearest_neighbors.get_container().reserve(k);

  double best_distance = std::numeric_limits<double>::max();

  Adapter<std::priority_queue< NeighborNode , std::vector<NeighborNode>, std::greater<NeighborNode> >> node_queue;
  node_queue.get_container().reserve(100);

  if (starting_node && starting_node->m_parent)
  {
    auto current_node = starting_node;

    // go to the leaf of this branch
    while (current_node)
    {
      // We arrived at a leaf, if it contains a single point, process as usual
      // If it contains several point, we use brute force to compute all 
      // distances in the leaf. This optimization helps reducing the number of
      // branchs in the tree, and is adapted from scipy ckdtree implementation.
      if (!current_node->m_indices.empty())
      {
        processLeaf(x, y, z, current_node, best_distance, k_nearest_neighbors, k);
        break;
      }

      const auto current_idx = current_node->m_idx;
      const auto current_dimension = current_node->m_dimension;

      // check if current node is a better match
      auto current_distance = squared_euclidian_distance_in_tree(x, y, z, current_node->m_idx);

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
      const double axis_distance = compute_axis_distance(current_dimension, x, y, z, current_idx);
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

    // Go back to root node
    current_node = starting_node->m_parent;
    KDNode* last_node = starting_node;
    while (current_node)
    {
      const auto current_idx = current_node->m_idx;
      const auto current_dimension = current_node->m_dimension;

      // check if current node is a better match
      auto current_distance = squared_euclidian_distance_in_tree(x, y, z, current_node->m_idx);

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
      const double axis_distance = compute_axis_distance(current_dimension, x, y, z, current_idx);
      const bool is_left = axis_distance < 0;
      const double squared_axis_distance = axis_distance*axis_distance;
      KDNode* other_branch_node = last_node == current_node->m_left_child  ? current_node->m_right_child : current_node->m_left_child;

      // Add the node on the other side of tree to the stack of node to be 
      // processed. The check verifying that neighbors can actually be on
      // this side is also done at the beginning of the loop, to avoid false 
      // negatives in case a smaller distance is found in this branch
      if (squared_axis_distance < best_distance)
      {
        node_queue.emplace(other_branch_node, squared_axis_distance);
      }
      last_node = current_node;
      current_node = current_node->m_parent;
    }
  }
  else
  {
    node_queue.emplace(m_root_node, 0);
  }

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
        processLeaf(x, y, z, current_node, best_distance, k_nearest_neighbors, k);
        break;
      }

      const auto current_idx = current_node->m_idx;
      const auto current_dimension = current_node->m_dimension;

      // check if current node is a better match
      auto current_distance = squared_euclidian_distance_in_tree(x, y, z, current_node->m_idx);

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
      const double axis_distance = compute_axis_distance(current_dimension, x, y, z, current_idx);
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

void KDTree::processLeaf(double x,
                         double y,
                         double z,
                         KDNode* node,
                         double& best_distance,
                         KNeighborList& k_nearest_neighbors, 
                         const unsigned int k)
{
  for (auto idx_it = node->m_indices.begin(); idx_it !=  node->m_indices.end(); idx_it++)
  {
    // check if current node is a better match
    auto current_distance = squared_euclidian_distance(x, y, z, 
                                      m_x[*idx_it], m_y[*idx_it], m_z[*idx_it]);
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

void KDTree::processLeafBall(const PointType& point, KDNode* node, std::vector<unsigned int>& neighbors, double squared_radius)
{
  for (auto idx_it = node->m_indices.begin(); idx_it !=  node->m_indices.end(); idx_it++)
  {
    PointType node_point = {m_x[*idx_it],
                            m_y[*idx_it],
                            m_z[*idx_it]};
    // check if current node is in the ball around the point
    if (squared_euclidian_distance(point, node_point) <= squared_radius)
    {
      neighbors.push_back(*idx_it);
    }

  }
}

KDNode* KDTree::build_tree()
{
  auto indexes = std::vector<unsigned int>(m_point_cloud.size());
  std::iota(indexes.begin(), indexes.end(), 0);

  // First split to find the root node
  KDNode* root_node = grow_tree(indexes.begin(), indexes.size(), 0, nullptr);

  return root_node;
}

}