/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

#include <array>

namespace ArborX
{
class Point
{
public:
  KOKKOS_INLINE_FUNCTION
  constexpr Point()
      : _coords{}
  {
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point(const std::array<double, 3> &coords)
      : _coords{coords}
  {
  }

  KOKKOS_INLINE_FUNCTION
  double &operator[](unsigned int i) { return _coords[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr double const &operator[](unsigned int i) const
  {
    return _coords[i];
  }

private:
  std::array<double, 3> _coords;
};
} // namespace ArborX

#endif
