/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_POINT_CLOUDS_HPP
#define ARBORX_POINT_CLOUDS_HPP

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <fstream>
#include <random>

enum class PointCloudType
{
  filled_box,
  hollow_box,
  filled_sphere,
  hollow_sphere
};

PointCloudType to_point_cloud_enum(std::string const &str)
{
  if (str == "filled_box")
    return PointCloudType::filled_box;
  if (str == "hollow_box")
    return PointCloudType::hollow_box;
  if (str == "filled_sphere")
    return PointCloudType::filled_sphere;
  if (str == "hollow_sphere")
    return PointCloudType::hollow_sphere;
  throw std::runtime_error(str +
                           " doesn't correspond to any known PointCloudType!");
}

template <typename DeviceType>
void filledBoxCloud(float const half_edge,
                    Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;

  GeneratorPool rand_pool(0);
  unsigned int const n = random_points.extent(0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_filledBoxCloud"),
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        auto random = [&rand_gen, half_edge]() {
          return Kokkos::rand<GeneratorType, float>::draw(rand_gen, -half_edge,
                                                          half_edge);
        };
        random_points(i) = {{random(), random(), random()}};
      });
}

template <typename DeviceType>
void hollowBoxCloud(float const half_edge,
                    Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;

  GeneratorPool rand_pool(0);
  unsigned int const n = random_points.extent(0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_filledBoxCloud"),
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        auto random = [&rand_gen, half_edge]() {
          return Kokkos::rand<GeneratorType, float>::draw(rand_gen, -half_edge,
                                                          half_edge);
        };

        unsigned int face = i % 6;
        switch (face)
        {
        case 0:
          random_points(i) = {{-half_edge, random(), random()}};
          break;
        case 1:
          random_points(i) = {{half_edge, random(), random()}};
          break;
        case 2:
          random_points(i) = {{random(), -half_edge, random()}};
          break;
        case 3:
          random_points(i) = {{random(), half_edge, random()}};
          break;
        case 4:
          random_points(i) = {{random(), random(), -half_edge}};
          break;
        case 5:
          random_points(i) = {{random(), random(), half_edge}};
          break;
        default:
          random_points(i) = {{0., 0., 0.}};
        }
      });
}

template <typename DeviceType>
void filledSphereCloud(float const radius,
                       Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;

  GeneratorPool rand_pool(0);
  unsigned int const n = random_points.extent(0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_filledBoxCloud"),
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        auto random = [&rand_gen, radius]() {
          return Kokkos::rand<GeneratorType, float>::draw(rand_gen, -radius,
                                                          radius);
        };
        bool point_accepted = false;
        while (!point_accepted)
        {
          double const x = random();
          double const y = random();
          double const z = random();

          // Only accept points that are in the sphere
          if (std::sqrt(x * x + y * y + z * z) <= radius)
          {
            random_points(i) = {{x, y, z}};
            point_accepted = true;
          }
        }
      });
}

template <typename DeviceType>
void hollowSphereCloud(double const radius,
                       Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;

  GeneratorPool rand_pool(0);
  unsigned int const n = random_points.extent(0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_filledBoxCloud"),
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        auto random = [&rand_gen, radius]() {
          return Kokkos::rand<GeneratorType, float>::draw(rand_gen, -radius,
                                                          radius);
        };
        double const x = random();
        double const y = random();
        double const z = random();
        double const norm = std::sqrt(x * x + y * y + z * z);

        random_points(i) = {
            {radius * x / norm, radius * y / norm, radius * z / norm}};
      });
}

template <typename DeviceType>
void generatePointCloud(PointCloudType const point_cloud_type,
                        double const length,
                        Kokkos::View<ArborX::Point *, DeviceType> random_points)
{
  switch (point_cloud_type)
  {
  case PointCloudType::filled_box:
    std::cout << "filledBox begin" << std::endl;
    filledBoxCloud(length, random_points);
    std::cout << "filledBox end" << std::endl;
    break;
  case PointCloudType::hollow_box:
    std::cout << "hollowBox begin" << std::endl;
    hollowBoxCloud(length, random_points);
    std::cout << "hollowBox end" << std::endl;
    break;
  case PointCloudType::filled_sphere:
    std::cout << "filledSphere begin" << std::endl;
    filledSphereCloud(length, random_points);
    std::cout << "filledSphere end" << std::endl;
    break;
  case PointCloudType::hollow_sphere:
    std::cout << "hollowSphere begin" << std::endl;
    hollowSphereCloud(length, random_points);
    std::cout << "hollowSphere end" << std::endl;
    break;
  default:
    throw ArborX::SearchException("not implemented");
  }
}

#endif
