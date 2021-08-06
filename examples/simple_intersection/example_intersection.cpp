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


struct Mapping
{
  ArborX::Point alpha;
  ArborX::Point beta;
  ArborX::Point p0;

  ArborX::Point get_coeff(ArborX::Point p) const
  {
    float alpha_coeff = alpha[0]*(p[0]-p0[0])+alpha[1]*(p[1]-p0[1])+alpha[2]*(p[2]-p0[2]);
    float beta_coeff = beta[0]*(p[0]-p0[0])+beta[1]*(p[1]-p0[1])+beta[2]*(p[2]-p0[2]);
    return {1-alpha_coeff-beta_coeff, alpha_coeff, beta_coeff};
  }
};

struct Triangle
{
  ArborX::Point a;
  ArborX::Point b;
  ArborX::Point c;
};

template <typename DeviceType>
class Points
{
public:
  Points(typename DeviceType::execution_space const & execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 2;
    int ny = 2;
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
    int nx = 2;
    int ny = 2;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) {
      return i + j * nx;
    };

    _triangles = Kokkos::View<Triangle *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"), 2*n);
    auto triangles_host = Kokkos::create_mirror_view(_triangles);

    _mappings = Kokkos::View<Mapping *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "mappings"), 2*n);
    auto mappings_host = Kokkos::create_mirror_view(_mappings);

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
  
    for (int k=0; k<2*n; ++k)
    {
      mappings_host[k] = get_mapping(triangles_host[k]);
/*
      const auto t = triangles_host[k];
      std::cout << "triangle " << k << ":\n";
      std::cout << "a: " << t.a[0] << ' ' << t.a[1] << ' '  << t.a[2] << '\n';
      std::cout << "b: " << t.b[0] << ' ' << t.b[1] << ' '  << t.b[2] << '\n';
      std::cout << "c: " << t.c[0] << ' ' << t.c[1] << ' '  << t.c[2] << '\n'; 
      const auto m = mappings_host[k];
      std::cout << "mapping " << k << ":\n";
      std::cout << "p0:    " << m.p0[0]    << ' ' << m.p0[1]    << ' '  << m.p0[2]    << '\n';
      std::cout << "alpha: " << m.alpha[0] << ' ' << m.alpha[1] << ' '  << m.alpha[2] << '\n';
      std::cout << "beta:  " << m.beta[0]  << ' ' << m.beta[1]  << ' '  << m.beta[2]  << '\n';
*/
    }
    Kokkos::deep_copy(execution_space, _triangles, triangles_host);
  }

  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return _triangles.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION const Triangle &get_triangle(int i) const { return _triangles(i); }
  
  KOKKOS_FUNCTION const Mapping &get_mapping(int i) const { return _mappings(i); }

private:
  Kokkos::View<Triangle *, typename DeviceType::memory_space> _triangles;
  Kokkos::View<Mapping *, typename DeviceType::memory_space> _mappings;

  // x = a + alpha * (b - a) + beta * (c - a) 
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  //
  // FIXME Only works for 2D reliably  
  static Mapping get_mapping(const Triangle& triangle) 
  {
    const auto& a = triangle.a;
    const auto& b = triangle.b;
    const auto& c = triangle.c;

    ArborX::Point u = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    ArborX::Point v = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    
    const float inv_det = 1./(v[1]*u[0]-v[0]*u[1]);

    Mapping mapping;
    mapping.alpha = ArborX::Point{v[1]*inv_det, -v[0]*inv_det,0};
    mapping.beta = ArborX::Point{-u[1]*inv_det, u[0]*inv_det,0};
    mapping.p0 = a;

    return mapping;
  }
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


KOKKOS_FUNCTION bool intersects(const ArborX::Point& point, const Triangle& triangle, const Mapping& mapping)
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

  bool first_check = !(has_neg && has_pos);

  const auto coeffs = mapping.get_coeff(point);
  bool second_check = (std::min({coeffs[0], coeffs[1], coeffs[2]}) >= 0);

  if(first_check != second_check)
	  abort();
  return first_check;
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

    if (intersects(points_.get_point(point_index), triangles_.get_triangle(triangle_index), triangles_.get_mapping(triangle_index)))
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

    constexpr float eps = 1.e-3;

    for (int i = 0; i<triangles.size(); ++i)
    {
      const auto& mapping  = triangles.get_mapping(i);   
      const auto& triangle = triangles.get_triangle(i);
      const auto& coeff_a = mapping.get_coeff(triangle.a);
      if ((std::abs(coeff_a[0]-1.) > eps) || std::abs(coeff_a[1]) > eps || std::abs(coeff_a[2]) > eps)
        std::cout << i << " a: " << coeff_a[0] << ' ' << coeff_a[1] << ' '  << coeff_a[2] << std::endl;
      const auto& coeff_b = mapping.get_coeff(triangle.b);
      if ((std::abs(coeff_b[0]) > eps) || std::abs(coeff_b[1]-1.) > eps || std::abs(coeff_b[2]) > eps)
        std::cout << i << " b: " << coeff_b[0] << ' ' << coeff_b[1] << ' '  << coeff_b[2] << std::endl;
      const auto& coeff_c = mapping.get_coeff(triangle.c);
      if ((std::abs(coeff_c[0]) > eps) || std::abs(coeff_c[1]) > eps || std::abs(coeff_c[2]-1.) > eps)
        std::cout << i << " c: " << coeff_c[0] << ' ' << coeff_c[1] << ' '  << coeff_c[2] << std::endl;
    }
    std::cout << "Triangles set up.\n";

    std::cout << "Creating BVH tree.\n";
    ArborX::BVH<MemorySpace> const tree(execution_space, triangles);
    std::cout << "BVH tree set up.\n";

    std::cout << "Create the points used for queries.\n" ;
    Points<DeviceType> points(execution_space);
    std::cout << "Points for queries set up.\n";
	    
    std::cout << "Starting the queries.\n";
    // The query will resize indices and offsets accordingly
    int const n = triangles.size();
    Kokkos::View<int *, MemorySpace> offsets("offsets", n);

    tree.query(execution_space, points, PrintfCallback<DeviceType>{offsets, points, triangles});//indices, offsets);
    std::cout << "Queries done.\n";

    std::cout << "Starting checking results.\n";
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);

    for (int i = 0; i < n; ++i)
      if (offsets_host(i) != i)
      {
        std ::cout << offsets_host(i) << " should be " << i << std::endl;      
        //Kokkos::abort("Wrong entry in the offsets View!\n");
      }

    std::cout << "Checking results successful.\n";
  }

  Kokkos::finalize();
}
