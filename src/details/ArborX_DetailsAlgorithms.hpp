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
#ifndef ARBORX_DETAILS_ALGORITHMS_HPP
#define ARBORX_DETAILS_ALGORITHMS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // min, max, isFinite
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Point const &l, Point const &r)
{
  for (int d = 0; d < 3; ++d)
    if (l[d] != r[d])
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Box const &l, Box const &r)
{
  return equals(l.minCorner(), r.minCorner()) &&
         equals(l.maxCorner(), r.maxCorner());
}

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Sphere const &l, Sphere const &r)
{
  return equals(l.centroid(), r.centroid()) && l.radius() == r.radius();
}

KOKKOS_INLINE_FUNCTION
bool isValid(Point const &p)
{
  using KokkosExt::isFinite;
  for (int d = 0; d < 3; ++d)
    if (!isFinite(p[d]))
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
bool isValid(Box const &b)
{
  return isValid(b.minCorner()) && isValid(b.maxCorner());
}

KOKKOS_INLINE_FUNCTION
bool isValid(Sphere const &s)
{
  using KokkosExt::isFinite;
  return isValid(s.centroid()) && isFinite(s.radius()) && (s.radius() >= 0.);
}

class DistanceReturnType
{
private:
  float sq = 0.;

public:
  // FIXME I do not like that I had to provide a default constructor to be
  // able to use in the PriorityQueue which holds a C array of
  // pair<DistanceReturnType, Node*>
  KOKKOS_FUNCTION DistanceReturnType() = default;
  // NOTE cannot be declared constexpr because abort() isn't
  KOKKOS_FUNCTION explicit DistanceReturnType(float v)
      : sq(v)
  {
    /*        if ( KokkosHelpers::isNan( sq ) || sq < 0 )
                Kokkos::abort( "Invalid arguemnt: DistanceReturnType constructor
       " "requires non-negative floating-point value" );*/
  }
  KOKKOS_INLINE_FUNCTION float to_float() const { return std::sqrt(sq); }
  KOKKOS_INLINE_FUNCTION bool operator<(DistanceReturnType const &rhs) const
  {
    return sq < rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator<(float const &rhs) const
  {
    return sq < rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator<(float const &lhs,
                                               DistanceReturnType const &rhs)
  {
    return lhs * lhs < rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator>(DistanceReturnType const &rhs) const
  {
    return sq > rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator>(float const &rhs) const
  {
    return sq > rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator>(float const &lhs,
                                               DistanceReturnType const &rhs)
  {
    return lhs * lhs > rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator==(DistanceReturnType const &rhs) const
  {
    return sq == rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator==(float const &rhs) const
  {
    return sq == rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator==(float const &lhs,
                                                DistanceReturnType const &rhs)
  {
    return lhs * lhs == rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator<=(DistanceReturnType const &rhs) const
  {
    return sq <= rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator<=(float const &rhs) const
  {
    return sq <= rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator<=(float const &lhs,
                                                DistanceReturnType const &rhs)
  {
    return lhs * lhs <= rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator>=(DistanceReturnType const &rhs) const
  {
    return sq >= rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator>=(float const &rhs) const
  {
    return sq >= rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator>=(float const &lhs,
                                                DistanceReturnType const &rhs)
  {
    return lhs * lhs >= rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator!=(DistanceReturnType const &rhs) const
  {
    return sq != rhs.sq;
  }
  KOKKOS_INLINE_FUNCTION bool operator!=(float const &rhs) const
  {
    return sq != rhs * rhs;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator!=(float const &lhs,
                                                DistanceReturnType const &rhs)
  {
    return lhs * lhs != rhs.sq;
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                DistanceReturnType const &distance)
{
  os << distance.to_float();
  return os;
}

// distance point-point
KOKKOS_INLINE_FUNCTION
DistanceReturnType distance(Point const &a, Point const &b)
{
  float distance_squared = 0.0;
  for (int d = 0; d < 3; ++d)
  {
    float tmp = b[d] - a[d];
    distance_squared += tmp * tmp;
  }
  return DistanceReturnType{distance_squared};
}

// distance point-box
KOKKOS_INLINE_FUNCTION
DistanceReturnType distance(Point const &point, Box const &box)
{
  Point projected_point;
  for (int d = 0; d < 3; ++d)
  {
    if (point[d] < box.minCorner()[d])
      projected_point[d] = box.minCorner()[d];
    else if (point[d] > box.maxCorner()[d])
      projected_point[d] = box.maxCorner()[d];
    else
      projected_point[d] = point[d];
  }
  return distance(point, projected_point);
}

// distance point-sphere
KOKKOS_INLINE_FUNCTION
DistanceReturnType distance(Point const &point, Sphere const &sphere)
{
  using KokkosExt::max;
  float const real_distance =
      max(distance(point, sphere.centroid()).to_float() - sphere.radius(), 0.f);
  return DistanceReturnType(real_distance * real_distance);
}

// expand an axis-aligned bounding box to include a point
KOKKOS_INLINE_FUNCTION
void expand(Box &box, Point const &point)
{
  using KokkosExt::max;
  using KokkosExt::min;
  for (int d = 0; d < 3; ++d)
  {
    box.minCorner()[d] = min(box.minCorner()[d], point[d]);
    box.maxCorner()[d] = max(box.maxCorner()[d], point[d]);
  }
}

// expand an axis-aligned bounding box to include another box
// NOTE: Box type is templated here to be able to use expand(box, box) in a
// Kokkos::parallel_reduce() in which case the arguments must be declared
// volatile.
template <typename BOX,
          typename = typename std::enable_if<std::is_same<
              typename std::remove_volatile<BOX>::type, Box>::value>::type>
KOKKOS_INLINE_FUNCTION void expand(BOX &box, BOX const &other)
{
  using KokkosExt::max;
  using KokkosExt::min;
  for (int d = 0; d < 3; ++d)
  {
    box.minCorner()[d] = min(box.minCorner()[d], other.minCorner()[d]);
    box.maxCorner()[d] = max(box.maxCorner()[d], other.maxCorner()[d]);
  }
}

// expand an axis-aligned bounding box to include a sphere
KOKKOS_INLINE_FUNCTION
void expand(Box &box, Sphere const &sphere)
{
  using KokkosExt::max;
  using KokkosExt::min;
  for (int d = 0; d < 3; ++d)
  {
    box.minCorner()[d] =
        min(box.minCorner()[d], sphere.centroid()[d] - sphere.radius());
    box.maxCorner()[d] =
        max(box.maxCorner()[d], sphere.centroid()[d] + sphere.radius());
  }
}

// check if two axis-aligned bounding boxes intersect
KOKKOS_INLINE_FUNCTION
constexpr bool intersects(Box const &box, Box const &other)
{
  for (int d = 0; d < 3; ++d)
    if (box.minCorner()[d] > other.maxCorner()[d] ||
        box.maxCorner()[d] < other.minCorner()[d])
      return false;
  return true;
}

KOKKOS_INLINE_FUNCTION
constexpr bool intersects(Point const &point, Box const &other)
{
  for (int d = 0; d < 3; ++d)
    if (point[d] > other.maxCorner()[d] || point[d] < other.minCorner()[d])
      return false;
  return true;
}

// check if a sphere intersects with an  axis-aligned bounding box
KOKKOS_INLINE_FUNCTION
bool intersects(Sphere const &sphere, Box const &box)
{
  return distance(sphere.centroid(), box) <= sphere.radius();
}

// calculate the centroid of a box
KOKKOS_INLINE_FUNCTION
void centroid(Box const &box, Point &c)
{
  for (int d = 0; d < 3; ++d)
    c[d] = (box.minCorner()[d] + box.maxCorner()[d]) / 2;
}

KOKKOS_INLINE_FUNCTION
void centroid(Point const &point, Point &c) { c = point; }

KOKKOS_INLINE_FUNCTION
void centroid(Sphere const &sphere, Point &c) { c = sphere.centroid(); }

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Point const &point) { return point; }

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Box const &box)
{
  Point c;
  for (int d = 0; d < 3; ++d)
    c[d] = (box.minCorner()[d] + box.maxCorner()[d]) / 2;
  return c;
}

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Sphere const &sphere) { return sphere.centroid(); }

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
KOKKOS_INLINE_FUNCTION
void translateAndScale(Point const &in, Point &out, Box const &ref)
{
  for (int d = 0; d < 3; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace Details
} // namespace ArborX

#endif
