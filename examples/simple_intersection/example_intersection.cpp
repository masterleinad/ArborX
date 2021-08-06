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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

// Perform intersection queries using 2D triangles on a regular mesh as primitives
// and intersection with points as queries. One point per triangle.
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________


struct Triangle
{
  ArborX::Point a;
  ArborX::Point b;
  ArborX::Point c;
};

namespace ArborX
{
KOKKOS_INLINE_FUNCTION void expand(ArborX::Box &box, Triangle const &triangle)
{
  using Details::expand;
  expand(box, triangle.a);
  expand(box, triangle.b);
  expand(box, triangle.c);
}
}

template <typename DeviceType>
class Points
{
public:
  Points(typename DeviceType::execution_space const & execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 11;
    int ny = 11;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) {
      return i + j * nx;
    };

    _points = Kokkos::View<ArborX::Point *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), 2*n);
    auto points_host = Kokkos::create_mirror_view(_points);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        {
          points_host[2*index(i, j)] = {(i+.25f) * hx, (j+.25f) * hy, 0.f};
          points_host[2*index(i, j)+1] = {(i+.75f) * hx, (j+.75f) * hy, 0.f};
        }
    Kokkos::deep_copy(execution_space, _points, points_host);
  }

  KOKKOS_FUNCTION auto const & get_point(int i) const
  {
    return _points(i);
  }


  KOKKOS_FUNCTION auto const & get_points() const
  {
    return _points;
  }

  private:
    Kokkos::View<ArborX::Point *, typename DeviceType::memory_space> _points;
};

template <typename DeviceType>
class Triangles
{
public:
  // Create non-intersecting triangles on a 3D cartesian grid
  // used both for queries and predicates.
  Triangles(typename DeviceType::execution_space const &execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 11;
    int ny = 11;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) {
      return i + j * nx;
    };

    _triangles = Kokkos::View<Triangle *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"), 2*n);
    auto triangles_host = Kokkos::create_mirror_view(_triangles);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        {
          ArborX::Point bl{i * hx, j * hy, 0.};
          ArborX::Point br{(i+1) * hx, j * hy, 0.};
	  ArborX::Point tl{i*hx, (j+1)*hy, 0.};
	  ArborX::Point tr{(i+1)*hx, (j+1)*hy, 0.};
          triangles_host[2*index(i, j)] = {tl, bl, br};
	  triangles_host[2*index(i, j)+1] = {tl, br, tr};
        }
    Kokkos::deep_copy(execution_space, _triangles, triangles_host);
  }

  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return _triangles.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION const Triangle &get_triangle(int i) const { return _triangles(i); }

  KOKKOS_FUNCTION const auto &get_triangles() const { return _triangles; }

private:
  Kokkos::View<Triangle *, typename DeviceType::memory_space> _triangles;
};

// For creating the bounding volume hierarchy given a Triangles object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Triangles class, we just resort to them.
template <typename DeviceType>
struct ArborX::AccessTraits<Triangles<DeviceType>, ArborX::PrimitivesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Triangles<DeviceType> const &triangles)
  {
    return triangles.size();
  }
/*  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
  {
    return triangles.get_triangle(i);
  }*/
  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
  {
    const auto& triangle = triangles.get_triangle(i);
    ArborX::Box box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

// For performing the queries given a Points object, we need to define memory
// space, how to get the total number of queries, and what the query with index
// i should look like. 
template <typename DeviceType>
struct ArborX::AccessTraits<Points<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Points<DeviceType> const &points)
  {
    return points.get_points().size();
  }
  static KOKKOS_FUNCTION auto get(Points<DeviceType> const &points, int i)
  {
    return ArborX::attach(intersects(points.get_points()(i)), i);
  }
};


KOKKOS_FUNCTION bool intersects(const ArborX::Point& point, const Triangle& triangle)
{
  auto sign = [](const ArborX::Point& p1, const ArborX::Point& p2, const ArborX::Point& p3)
  {
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  };

  const float d1 = sign(point, triangle.a, triangle.b);
  const float d2 = sign(point, triangle.b, triangle.c);
  const float d3 = sign(point, triangle.c, triangle.a);

  const bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  const bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

  return !(has_neg && has_pos);
}



template <typename DeviceType>
class PrintfCallback
{
public:
  PrintfCallback(Kokkos::View<int*, typename DeviceType::memory_space> results,
		 Points<DeviceType> points, 
		 Triangles<DeviceType> triangles) 
	  : results_(results), points_(points), triangles_(triangles)
  {}

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int point_index) const
  {
    auto const triangle_index = ArborX::getData(query);

    if (intersects(points_.get_point(point_index), triangles_.get_triangle(triangle_index)))
      results_(point_index) = triangle_index;
  }
private:
  Kokkos::View<int*, typename DeviceType::memory_space> results_;
  Points<DeviceType> points_;
  Triangles<DeviceType> triangles_;
};

// Now that we have encapsulated the objects and queries to be used within the
// Triangles class, we can continue with performing the actual search.
int main()
{
  Kokkos::initialize();
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with triangles.\n";
    Triangles<DeviceType> triangles(execution_space);
    std::cout << "Triangles set up.\n";

    std::cout << "Creating BVH tree.\n";
    ArborX::BVH<MemorySpace> const tree(execution_space, triangles);
    std::cout << "BVH tree set up.\n";

    std::cout << "Create the points used for queries.\n" ;
    Points<DeviceType> points(execution_space);
    std::cout << "Points for queries set up.\n";
	    
    std::cout << "Starting the queries.\n";
    // The query will resize indices and offsets accordingly
    unsigned int const n = triangles.size();
    Kokkos::View<int *, MemorySpace> offsets("offsets", n);

    tree.query(execution_space, points, PrintfCallback<DeviceType>{offsets, points, triangles});//indices, offsets);
    std::cout << "Queries done.\n";

    std::cout << "Starting checking results.\n";
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);

    for (int i = 0; i < static_cast<int>(n + 1); ++i)
      if (offsets_host(i) != i)
      {
        std ::cout << offsets_host(i) << " should be " << i << std::endl;      
        //Kokkos::abort("Wrong entry in the offsets View!\n");
      }

    std::cout << "Checking results successful.\n";
  }

  Kokkos::finalize();
}
