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

namespace cars_filter
{

/**
 * @class Image
 * @brief Image interface the providing pixel accessors from a data pointer and
 * image dimensions. This class does not manage the lifetime of the underlying 
 * pointer.
 *
 */
template<typename T>
class Image
{
public:

  /**
   * getter to the underlying pointer
   */
  T* getData()
  {
    return m_data;
  }

  /**
   * getter to element at index (row, col)
   */
  T& get(unsigned int row, unsigned int col)
  {
    return m_data[col + m_num_cols * row];
  }

  /**
   * getter to element at index (row, col) (read-only)
   */
  const T& get(unsigned int row, unsigned int col) const
  {
    return m_data[col + m_num_cols * row];
  }

  /**
   * Number of pixel in the image
   */
  unsigned int size() const
  {
    return m_num_rows * m_num_cols;
  }

  /**
   * Getter to the number of rows
   */
  unsigned int number_of_rows() const
  {
    return m_num_rows;
  }

  /**
   * Getter to the number of columns
   */
  unsigned int number_of_cols() const
  {
    return m_num_cols;
  }


  /**
   * Constructor
   *
   * @param data pointer to input data
   * @param num_rows number of row of the image
   * @param num_cols number of columns of the image
   */
  Image(T* data,
        unsigned int num_rows,
        unsigned int num_cols)
          : m_data(data), m_num_rows(num_rows), m_num_cols(num_cols)
  {
  }

protected:
  T* m_data;

private:
  unsigned int m_num_rows;
  unsigned int m_num_cols;
};


/**
 * @class InMemoryImage
 * @brief Class implementing an image in memory
 *
 */
template<typename T>
class InMemoryImage: public Image<T>
{
public:

  /**
   * Constructor from number of rows and cols, allocate num_rows*num_cols elements
   *
   * @param num_rows number of row of the image
   * @param num_cols number of columns of the image
   */
  InMemoryImage(unsigned int num_rows, unsigned int num_cols)
    : Image<T>(new double[num_rows*num_cols]{}, num_rows, num_cols)
  {
  }

  /**
   * Destructor, free the allocated memory
   */
  ~InMemoryImage()
  {
    if (this->m_data)
    {
      delete[] this->m_data;
    }
  }
};


}