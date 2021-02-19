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
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
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

std::vector<ArborX::Point> loadNGSIMData(std::string const &filename,
                                         bool binary)
{
  assert(!binary);
  std::cout << "Trying to open " << filename << " assuming NGSIM data.\n";
  std::ifstream file(filename);
  assert(file.good());

  std::string thisWord, line;

  std::vector<ArborX::Point> v;
  int n_points = 0;

  // ignore first line that contains the descriptions
  getline(file, thisWord);
  while (file.good() && n_points < 1000000)
  {
    getline(file, line);
    std::stringstream ss(line);
    // GVehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y
    for (int i = 0; i < 6; ++i)
      getline(ss, thisWord, ',');
    // Global_X,Global_Y
    getline(ss, thisWord, ',');
    float longitude = stof(thisWord);
    getline(ss, thisWord, ',');
    float latitude = stof(thisWord);
    v.emplace_back(longitude, latitude, 0.f);
    // v_length,v_Width,v_Class,v_Vel,v_Acc,Lane_ID,O_Zone,D_Zone,Int_ID,Section_ID,Direction,Movement,Preceding,Following,Space_Headway,Time_Headway,Location
    for (int i = 0; i < 16; ++i)
      getline(ss, thisWord, ',');
    getline(ss, thisWord, ',');
    ++n_points;
  }
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;
  return v;
}

std::vector<ArborX::Point> loadTaxiPortoData(std::string const &filename,
                                             bool binary)
{
  assert(!binary);
  std::cout << "Trying to open " << filename << " assuming TaxiPorto data.\n";
  FILE *fp_data = fopen(filename.c_str(), "rb");
  assert(fp_data);
  char line[100000];

  // This function reads and segments trajectories in dataset in the following
  // format: The first line indicates number of variables per point (I'm ignoring
  // that and assuming 2) The second line indicates total trajectories in file
  // (I'm ignoring that and observing how many are there by reading them). All
  // lines that follow contains a trajectory separated by new line. The first
  // number in the trajectory is the number of points followed by location points
  // separated by spaces

  std::vector<float> longitudes;
  std::vector<float> latitudes;
  std::vector<ArborX::Point> v;

  int lineNo = -1;
  int wordNo = 0;
  int lonlatno = 100;

  float thisWord;
  while (fgets(line, sizeof(line), fp_data))
  {
    // std::cout << line << std::endl;
    if (lineNo > -1)
    {
      char *pch;
      char *end_str;
      wordNo = 0;
      lonlatno = 0;
      pch = strtok_r(line, "\"[", &end_str);
      while (pch != nullptr)
      {
        if (wordNo > 0)
        {
          char *pch2;
          char *end_str2;

          pch2 = strtok_r(pch, ",", &end_str2);

          if (strcmp(pch2, "]") < 0 && lonlatno < 255)
          {

            thisWord = atof(pch2);

            if (thisWord != 0.00000)
            {
              if (thisWord > -9 && thisWord < -7)
              {
                longitudes.push_back(thisWord);
                // printf("lon %f",thisWord);
                pch2 = strtok_r(nullptr, ",", &end_str2);
                thisWord = atof(pch2);
                if (thisWord < 42 && thisWord > 40)
                {
                  latitudes.push_back(thisWord);
                  // printf(" lat %f\n",thisWord);

                  lonlatno++;
                }
                else
                {
                  longitudes.pop_back();
                }
              }
            }
          }
        }
        pch = strtok_r(nullptr, "[", &end_str);
        wordNo++;
      }
      // printf("num lonlat were %d x 2\n",lonlatno);
    }
    lineNo++;
    if (lonlatno <= 0)
    {
      lineNo--;
    }

    // printf("Line %d\n",lineNo);
  }
  fclose(fp_data);

  int num_points = longitudes.size();
  assert(longitudes.size() == latitudes.size());
  v.reserve(num_points);
  for (unsigned int i = 0; i < num_points; ++i)
    v.emplace_back(longitudes[i], latitudes[i], 0.f);

  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

std::vector<ArborX::Point> load3dSpatialNetworkData(std::string const &filename,
                                                    bool binary)
{
  assert(!binary);
  std::cout << "Trying to open " << filename
            << " assuming 3dSpatialNetwork data.\n";
  std::ifstream file(filename);
  assert(file.good());

  std::vector<ArborX::Point> v;

  std::string thisWord;
  while (file.good())
  {
    getline(file, thisWord, ',');
    getline(file, thisWord, ',');
    float longitude = stof(thisWord);
    getline(file, thisWord, ',');
    float latitude = stof(thisWord);
    v.emplace_back(longitude, latitude, 0.f);
  }
  // really?????
  // lon_ptr.pop_back();
  // lat_ptr.pop_back();
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    std::string const &reader_type, bool binary)
{
  std::cout << "in reader" << std::endl;
  if (reader_type == "arborx")
    return loadArborXData(filename, binary);
  else if (reader_type == "NGSIM")
    return loadNGSIMData(filename, binary);
  else if (reader_type == "TaxiPorto")
    return loadTaxiPortoData(filename, binary);
  else if (reader_type == "3dSpatialNetwork")
    return load3dSpatialNetworkData(filename, binary);
  else
    throw std::runtime_error("Unknown reader type: \"" + reader_type + "\"");
}
