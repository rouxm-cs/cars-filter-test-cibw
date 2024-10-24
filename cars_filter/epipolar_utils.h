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

#include "image.h"

// points and distances
#include "KDTree.h"

namespace cars_filter
{

/*
* Check if a point is NaN
* 
* \param point Input point
* \return true if any coordinate of the point is nan
*
*/
inline bool isnan(const PointType& point)
{
  return std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]);
}

inline double squared_euclidian_distance(double x1, double y1, double z1, double x2, double y2, double z2)
{
  return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2); 
}

inline double squared_euclidian_distance(const PointType& point1, const PointType& point2) 
{
  return squared_euclidian_distance(point1[0],
                            point1[1],
                            point1[2],
                            point2[0],
                            point2[1],
                            point2[2]);
}

/*
* Find neighbors of point, in the sense of the euclian distance, inside a search
* window of the epipolar window
* 
*/
double epipolar_knn(const Image<double>& x_coords,
                    const Image<double>& y_coords,
                    const Image<double>& z_coords,
                    unsigned int ref_row,
                    unsigned int ref_col,
                    unsigned int k,
                    unsigned int half_window_size);


/*
* Find all points in a search window around the input point in epipolar geometry
* 
* \param x_coords Image in epipolar geometry containing the x triangulated coordinates
* \param y_coords Image in epipolar geometry containing the y triangulated coordinates
* \param z_coords Image in epipolar geometry containing the z triangulated coordinates
* \param ref_row row of the point of interest
* \param ref_col column of the point of interest
* \param radius distance threshold defining if a point is in the neighborhood or not
* \param half_window_size half size of the epipolar search window (in rows and columns) 
* 
*/
std::vector<std::pair<unsigned int, unsigned int>> epipolar_neighbors_in_ball(
                                              const Image<double>& x_coords,
                                              const Image<double>& y_coords,
                                              const Image<double>& z_coords,
                                              unsigned int ref_row,
                                              unsigned int ref_col,
                                              double radius,
                                              unsigned int half_window_size);

} // namespace cars_filter