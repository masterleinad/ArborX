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

#include <ArborX_BruteForce.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <unistd.h>

struct Dummy {
  int count;
};

using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

template <> struct ArborX::AccessTraits<Dummy, ArborX::PrimitivesTag> {
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy const &d) { return d.count; }
  static KOKKOS_FUNCTION Point get(Dummy const &, size_type i) {
    return {{(float)i, (float)i, (float)i}};
  }
};

template <> struct ArborX::AccessTraits<Dummy, ArborX::PredicatesTag> {
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy const &d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy const &, size_type i) {
    return attach(intersects(Sphere{{{(float)i, (float)i, (float)i}},
                                    (float) i}),
                  i);
  }
};

struct PrintfCallback {
  Kokkos::View<int, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> c_;
  PrintfCallback() : c_{"counter"} {}
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i,
                                  OutputFunctor const &out) const {
    int const j = getData(predicate);
    printf("%d callback (%d,%d)\n", ++c_(), i, j);
    out(i);
  }
};

template <typename T, typename... P>
std::vector<T> view2vec(Kokkos::View<T *, P...> view) {
  std::vector<T> vec(view.size());
  Kokkos::deep_copy(Kokkos::View<T *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        vec.data(), vec.size()),
                    view);
  return vec;
}

template <typename OutputView, typename OffsetView>
void print(OutputView const out, OffsetView const offset) {
  int const n_queries = offset.extent(0) - 1;

  auto const h_out = view2vec(out);
  auto const h_offset = view2vec(offset);
  int count = 0;

  for (int j = 0; j < n_queries; ++j)
    for (int k = h_offset[j]; k < h_offset[j + 1]; ++k)
      printf("%d result (%d, %d)\n", ++count, h_out[k], j);
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int nqueries = 5, nprimitives = 5, nrepeats = 1;
  int c;
  while ((c = getopt(argc, argv, "p:q:r:")) != -1)
    switch (c) {
    case 'p':
      nprimitives = atoi(optarg);
      break;
    case 'q':
      nqueries = atoi(optarg);
      break;
    case 'r':
      nrepeats = atoi(optarg);
      break;
    case '?':
      fprintf(stderr, "Usage: %s [-p <count> -q <count>]\n", argv[0]);
      return 1;
    default:
      abort();
    }

  printf("Primitives: %d\n", nprimitives);
  printf("Predicates: %d\n", nqueries);

  ExecutionSpace space{};
  Dummy primitives{nprimitives};
  Dummy predicates{nqueries};

  for (int i = 0; i < nrepeats; i++) {
    int out_count;
    // printf("Bounding Volume Hierarchy\n");
    {
      Kokkos::Timer timer;
      ArborX::BoundingVolumeHierarchy<MemorySpace> bvh{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices_ref", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset_ref", 0);
      bvh.query(
          space, predicates, ArborX::Details::DefaultCallback{},
          indices, offset,
          ArborX::Experimental::TraversalPolicy{}.setPredicateSorting(true));

      double time = timer.seconds();
      if (i == 0)
        printf("Collisions: %.5f\n",
               (float)(indices.extent(0)) / (nprimitives * nqueries));
      printf("Time BVH: %lf\n", time);
      out_count = indices.extent(0);
    }

    // printf("Brute Force\n");
    {
      Kokkos::Timer timer;
      ArborX::BruteForce<MemorySpace> brute{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset", 0);
      brute.query(space, predicates,
                  ArborX::Details::DefaultCallback{}, indices,
                  offset);

      double time = timer.seconds();
      printf("Time BF: %lf\n", time);
      ARBORX_ASSERT(out_count == indices.extent(0));
    }
  }
  return 0;
}
