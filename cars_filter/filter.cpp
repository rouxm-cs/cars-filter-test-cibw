
#include <pybind11/pybind11.h>
#include <iostream>


void pyOutlierFiltering()
{
  std::cout << "pyOutlierFiltering" << std::endl;
}


// wrap as Python module
PYBIND11_MODULE(outlier_filter, m)
{
  m.doc() = "filter";
  m.def("pc_outlier_filtering", &pyOutlierFiltering, "Filter outliers from point cloud");
}