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

#ifndef ARBORX_DETAILS_POINT_HPP
#define ARBORX_DETAILS_POINT_HPP

#include <Kokkos_Macros.hpp>

#include <utility>

namespace ArborX
{
class Point
{
private:
  struct Data
  {
    constexpr Data() noexcept : coords{0.,0.,0.}{}

    constexpr Data(const std::initializer_list<double> vals) noexcept :
    coords {vals.begin()[0], vals.begin()[1], vals.begin()[2]}
    {}

    double coords[3];
  } _data = {};

public:
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Point() noexcept = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Point(Data data)
      : _data{std::move(data)}
  {
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point(double x, double y, double z)
      : _data{{x, y, z}}
  {
  }

  KOKKOS_INLINE_FUNCTION
  constexpr double &operator[](unsigned int i) { return _data.coords[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr const double &operator[](unsigned int i) const
  {
    return _data.coords[i];
  }

  KOKKOS_INLINE_FUNCTION
  double volatile &operator[](unsigned int i) volatile
  {
    return _data.coords[i];
  }

  KOKKOS_INLINE_FUNCTION
  double const volatile &operator[](unsigned int i) const volatile
  {
    return _data.coords[i];
  }
};
} // namespace ArborX

#endif
