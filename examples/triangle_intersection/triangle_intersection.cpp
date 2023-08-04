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
#include <ArborX_HyperTriangle.hpp>

#include <Kokkos_Core.hpp>

// Perform intersection queries using 2D triangles on a regular mesh as
// primitives and intersection with points as queries. One point per triangle.
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
  ArborX::ExperimentalHyperGeometry::Point<2> alpha;
  ArborX::ExperimentalHyperGeometry::Point<2> beta;
  ArborX::ExperimentalHyperGeometry::Point<2> p0;

  ArborX::Point get_coeff(ArborX::ExperimentalHyperGeometry::Point<2> p) const
  {
    float alpha_coeff = alpha[0] * (p[0] - p0[0]) + alpha[1] * (p[1] - p0[1]);
    float beta_coeff = beta[0] * (p[0] - p0[0]) + beta[1] * (p[1] - p0[1]);
    return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
  }

  // x = a + alpha * (b - a) + beta * (c - a)
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  //
  // FIXME Only works for 2D reliably
  void compute(ArborX::ExperimentalHyperGeometry::Triangle<2> const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    ArborX::ExperimentalHyperGeometry::Point<2> u = {b[0] - a[0], b[1] - a[1]};
    ArborX::ExperimentalHyperGeometry::Point<2> v = {c[0] - a[0], c[1] - a[1]};

    float const inv_det = 1. / (v[1] * u[0] - v[0] * u[1]);

    alpha = ArborX::ExperimentalHyperGeometry::Point<2>{v[1] * inv_det,
                                                        -v[0] * inv_det};
    beta = ArborX::ExperimentalHyperGeometry::Point<2>{-u[1] * inv_det,
                                                       u[0] * inv_det};
    p0 = a;
  }

  ArborX::ExperimentalHyperGeometry::Triangle<2> get_triangle() const
  {
    float const inv_det = 1. / (alpha[0] * beta[1] - alpha[1] * beta[0]);
    ArborX::ExperimentalHyperGeometry::Point<2> a = p0;
    ArborX::ExperimentalHyperGeometry::Point<2> b = {
        {p0[0] + inv_det * beta[1], p0[1] - inv_det * beta[0]}};
    ArborX::ExperimentalHyperGeometry::Point<2> c = {
        {p0[0] - inv_det * alpha[1], p0[1] + inv_det * alpha[0]}};
    return {a, b, c};
  }
};

template <typename DeviceType>
class Points
{
public:
  Points(typename DeviceType::execution_space const &execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 101;
    int ny = 101;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) { return i + j * nx; };

    points_ = Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> *,
                           typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), 2 * n);
    auto points_host = Kokkos::create_mirror_view(points_);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
        points_host[2 * index(i, j)] = {(i + .25f) * hx, (j + .25f) * hy};
        points_host[2 * index(i, j) + 1] = {(i + .75f) * hx, (j + .75f) * hy};
      }
    Kokkos::deep_copy(execution_space, points_, points_host);
  }

  KOKKOS_FUNCTION auto const &get_point(int i) const { return points_(i); }

  KOKKOS_FUNCTION auto size() const { return points_.size(); }

private:
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<2> *,
               typename DeviceType::memory_space>
      points_;
};

template <typename DeviceType>
class Triangles
{
public:
  // Create non-intersecting triangles on a 2D cartesian grid
  // used both for queries and predicates.
  Triangles(typename DeviceType::execution_space const &execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 101;
    int ny = 101;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) { return i + j * nx; };

    triangles_ = Kokkos::View<ArborX::ExperimentalHyperGeometry::Triangle<2> *,
                              typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"), 2 * n);
    auto triangles_host = Kokkos::create_mirror_view(triangles_);

    mappings_ = Kokkos::View<Mapping *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "mappings"), 2 * n);
    auto mappings_host = Kokkos::create_mirror_view(mappings_);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
        ArborX::ExperimentalHyperGeometry::Point<2> bl{i * hx, j * hy};
        ArborX::ExperimentalHyperGeometry::Point<2> br{(i + 1) * hx, j * hy};
        ArborX::ExperimentalHyperGeometry::Point<2> tl{i * hx, (j + 1) * hy};
        ArborX::ExperimentalHyperGeometry::Point<2> tr{(i + 1) * hx,
                                                       (j + 1) * hy};

        triangles_host[2 * index(i, j)] = {tl, bl, br};
        triangles_host[2 * index(i, j) + 1] = {tl, br, tr};
      }

    for (int k = 0; k < 2 * n; ++k)
    {
      mappings_host[k].compute(triangles_host[k]);

      ArborX::ExperimentalHyperGeometry::Triangle<2> recover_triangle =
          mappings_host[k].get_triangle();

      for (unsigned int i = 0; i < 2; ++i)
        if (std::abs(triangles_host[k].a[i] - recover_triangle.a[i]) > 1.e-3)
          abort();

      for (unsigned int i = 0; i < 2; ++i)
        if (std::abs(triangles_host[k].b[i] - recover_triangle.b[i]) > 1.e-3)
          abort();

      for (unsigned int i = 0; i < 2; ++i)
        if (std::abs(triangles_host[k].c[i] - recover_triangle.c[i]) > 1.e-3)
          abort();
    }
    Kokkos::deep_copy(execution_space, triangles_, triangles_host);
  }

  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return triangles_.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION ArborX::ExperimentalHyperGeometry::Triangle<2> const &
  get_triangle(int i) const
  {
    return triangles_(i);
  }

  KOKKOS_FUNCTION Mapping const &get_mapping(int i) const
  {
    return mappings_(i);
  }

private:
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Triangle<2> *,
               typename DeviceType::memory_space>
      triangles_;
  Kokkos::View<Mapping *, typename DeviceType::memory_space> mappings_;
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
  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
  {
    auto const &triangle = triangles.get_triangle(i);
    ArborX::ExperimentalHyperGeometry::Box<2> box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

template <typename DeviceType>
class TriangleIntersectionCallback
{
public:
  TriangleIntersectionCallback(Triangles<DeviceType> triangles)
      : triangles_(triangles)
  {}

  template <typename Query>
  KOKKOS_FUNCTION void operator()(
      Query const &query,
      ArborX::Details::PairIndexVolume<
          ArborX::ExperimentalHyperGeometry::Box<2>> const &predicate) const
  {
    ArborX::ExperimentalHyperGeometry::Point<2> const &point =
        getGeometry(getPredicate(query));
    auto const &attachment = ArborX::getData(query);

    auto const &triangle = triangles_.get_triangle(predicate.index);

    auto const coeffs =
        triangles_.get_mapping(predicate.index).get_coeff(point);
    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;

    if (intersects)
    {
      attachment.triangle_index = predicate.index;
      attachment.coeffs = coeffs;
    }
  }

private:
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

    for (int i = 0; i < triangles.size(); ++i)
    {
      auto const &mapping = triangles.get_mapping(i);
      auto const &triangle = triangles.get_triangle(i);
      auto const &coeff_a = mapping.get_coeff(triangle.a);
      if ((std::abs(coeff_a[0] - 1.) > eps) || std::abs(coeff_a[1]) > eps ||
          std::abs(coeff_a[2]) > eps)
        std::cout << i << " a: " << coeff_a[0] << ' ' << coeff_a[1] << ' '
                  << coeff_a[2] << std::endl;
      auto const &coeff_b = mapping.get_coeff(triangle.b);
      if ((std::abs(coeff_b[0]) > eps) || std::abs(coeff_b[1] - 1.) > eps ||
          std::abs(coeff_b[2]) > eps)
        std::cout << i << " b: " << coeff_b[0] << ' ' << coeff_b[1] << ' '
                  << coeff_b[2] << std::endl;
      auto const &coeff_c = mapping.get_coeff(triangle.c);
      if ((std::abs(coeff_c[0]) > eps) || std::abs(coeff_c[1]) > eps ||
          std::abs(coeff_c[2] - 1.) > eps)
        std::cout << i << " c: " << coeff_c[0] << ' ' << coeff_c[1] << ' '
                  << coeff_c[2] << std::endl;
    }
    std::cout << "Triangles set up.\n";

    std::cout << "Creating BVH tree.\n";
    ArborX::BasicBoundingVolumeHierarchy<
        MemorySpace, ArborX::Details::PairIndexVolume<
                         ArborX::ExperimentalHyperGeometry::Box<2>>> const
        tree(execution_space, triangles);
    std::cout << "BVH tree set up.\n";

    std::cout << "Create the points used for queries.\n";
    Points<DeviceType> points(execution_space);
    std::cout << "Points for queries set up.\n";

    std::cout << "Starting the queries.\n";
    int const n = points.size();
    Kokkos::View<int *, MemorySpace> offsets("offsets", n);
    Kokkos::View<ArborX::Point *, MemorySpace> coefficients("coefficients", n);

    struct Dummy
    {};

    struct Attachment
    {
      int &triangle_index;
      ArborX::Point &coeffs;
    };

    ArborX::Details::TreeTraversal<
        decltype(tree), Dummy, TriangleIntersectionCallback<DeviceType>,
        ArborX::Details::SpatialPredicateTag,
        decltype(ArborX::attach(
            ArborX::intersects(ArborX::ExperimentalHyperGeometry::Point<2>{}),
            std::declval<Attachment>()))>
        tree_traversal(tree,
                       TriangleIntersectionCallback<DeviceType>{triangles});

    std::cout << "n: " << n << std::endl;

    Kokkos::parallel_for(
        "ArborX::TreeTraversal::spatial",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          tree_traversal.search(
              ArborX::attach(ArborX::intersects(points.get_point(i)),
                             Attachment{offsets(i), coefficients(i)}));
        });

    std::cout << "Queries done.\n";

    std::cout << "Starting checking results.\n";
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto coeffs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coefficients);

    for (int i = 0; i < n; ++i)
    {
      if (offsets_host(i) != i)
      {
        std::cout << offsets_host(i) << " should be " << i << std::endl;
      }
      auto const &c = coeffs_host(i);
      auto const &t = triangles.get_triangle(offsets_host(i));
      auto const &p_h = points.get_point(i);
      auto const p = ArborX::ExperimentalHyperGeometry::Point<2>{
          c[0] * t.a[0] + c[1] * t.b[0] + c[2] * t.c[0],
          c[0] * t.a[1] + c[1] * t.b[1] + c[2] * t.c[1]};
      if ((std::abs(p[0] - p_h[0]) > eps) || std::abs(p[1] - p_h[1]) > eps)
      {
        std::cout << "coeffs for point " << i << " are wrong!\n";
      }
    }

    std::cout << "Checking results successful.\n";
  }

  Kokkos::finalize();
}
