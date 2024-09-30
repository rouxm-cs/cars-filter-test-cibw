
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>


// debug
#include <chrono>
#include <thread>

namespace py = pybind11;


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

class KDTree
{
public:
  KDTree(const PointCloud& point_cloud): m_point_cloud(point_cloud)
  {
    m_nodes.reserve(point_cloud.size());
    build_tree();
  }


private:

  void build_tree()
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

    return &node;
  }

  PointCloud m_point_cloud;
  std::vector<KDNode> m_nodes;
};


void statistical_filtering(double* x_coords, double* y_coords, double* z_coords, unsigned int num_elem)
{
  auto input_point_cloud = PointCloud(x_coords, y_coords, z_coords, num_elem);

  auto tree = KDTree(input_point_cloud);

  std::cout << "statistical_filtering not implemented" << std::endl;
}

void pyOutlierFiltering(py::array_t<double,
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
  statistical_filtering(x_coords, y_coords, z_coords,  info.shape[1]);

}


// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_outlier_filtering", &pyOutlierFiltering, "Filter outliers from point cloud");
}