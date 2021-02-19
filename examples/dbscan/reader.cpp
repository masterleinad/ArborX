/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "reader.hpp"

#include <ArborX_Exception.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

std::vector<ArborX::Point> loadArborXData(std::string const &filename,
                                          bool binary = false)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  if (!binary)
  {
    input >> num_points;

    x.reserve(num_points);
    y.reserve(num_points);
    z.reserve(num_points);

    auto read_float = [&input]() {
      return *(std::istream_iterator<float>(input));
    };
    std::generate_n(std::back_inserter(x), num_points, read_float);
    std::generate_n(std::back_inserter(y), num_points, read_float);
    std::generate_n(std::back_inserter(z), num_points, read_float);
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);
    input.read(reinterpret_cast<char *>(x.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(y.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(z.data()), num_points * sizeof(float));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  std::vector<ArborX::Point> v(num_points);
  for (int i = 0; i < num_points; i++)
  {
    v[i] = {x[i], y[i], z[i]};
  }

  return v;
}

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    std::string const &reader_type, bool binary)
{
  if (reader_type == "arborx")
    return loadArborXData(filename, binary);
  else
    throw std::runtime_error("Unknown reader type: \"" + reader_type + "\"");
}
