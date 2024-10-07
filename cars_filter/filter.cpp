
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <queue>

#include <filesystem>

// debug
#include <chrono>
#include <thread>

namespace py = pybind11;

// TODO remove debug: recursion counter
static int num_recur = 0;

using PointType = std::array<double, 3>;

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
  double euclidian_distance(double x1, double y1, double z1, double x2, double y2, double z2)
  {
    return std::sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)); 
  }

  double euclidian_distance(double x, double y, double z, KDNode& node)
  {
    return euclidian_distance(x,
                              y,
                              z,
                              m_point_cloud.m_x[node.m_idx],
                              m_point_cloud.m_y[node.m_idx],
                              m_point_cloud.m_z[node.m_idx]);
  }

  double euclidian_distance(const PointType& point, const KDNode& node)
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



std::vector<unsigned int> statistical_filtering(double* x_coords,
                                                double* y_coords,
                                                double* z_coords,
                                                unsigned int num_elem)
{
  auto input_point_cloud = PointCloud(x_coords, y_coords, z_coords, num_elem);

  auto tree = KDTree(input_point_cloud);

  //tree.findNN(809438.52, 6304954.180000001, 55.53999999999999);
  //tree.findNN(809438.94, 6304941.72, 51.92999999999999);
  auto neighbor = tree.findNN(809438.94, 6304941.72, 50.92999999999999);

  std::cout << "-----------------------------------------" << std::endl;

  int k = 100;
  double dev_factor = 1.;

  auto k_neighbors = tree.findKNN(809438.94, 6304941.72, 51.92999999999999, k);

  //tree.findNN(809438.52, 6304954.180000001, 55.53999999999999);

  std::vector<double> mean_distances;
  mean_distances.reserve(num_elem);

  for (int i=0; i< input_point_cloud.size(); i++)
  {
    auto k_neighbors = tree.findKNN(input_point_cloud.m_x[i], 
                 input_point_cloud.m_y[i],
                 input_point_cloud.m_z[i],
                 k);
    double acc =0;
    for (const auto& elem : k_neighbors)
    {
      acc += elem.distance;
    }
    mean_distances.push_back(acc/k);
  }


  double dist_thresh;

  bool use_median = false;
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

  return result;
}

py::list pyOutlierFiltering(py::array_t<double,
          py::array::c_style | py::array::forcecast> points)
{
  /* Request a buffer descriptor from Python */
  py::buffer_info info = points.request();
  auto x_coords = static_cast<double *>(info.ptr);

  // TODO: remove debug info
  std::cout << "info.itemsize " << info.itemsize << std::endl;
  std::cout << "info.format " << info.format << std::endl;
  std::cout << "info.ndim " << info.ndim << std::endl;

  for (const auto& elem: info.shape)
  {
    std::cout << "shape " << elem << std::endl;
  }
  for (const auto& elem: info.strides)
  {
    std::cout << "strides " << elem << std::endl;
  }

  auto y_coords = x_coords + info.shape[1];
  auto z_coords = y_coords + info.shape[1];
  auto result = statistical_filtering(x_coords, y_coords, z_coords, info.shape[1]);

  // Copy C++ vector to Python list
  // As result has variable length, I am not sure if the copy can be avoided here in pyBind
  return py::cast(result);
}


// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_outlier_filtering", &pyOutlierFiltering, "Filter outliers from point cloud");
}