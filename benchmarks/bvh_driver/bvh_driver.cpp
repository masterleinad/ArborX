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

#include <ArborX_BoostRTreeHelpers.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

#ifdef ARBORX_PERFORMANCE_TESTING
#include <mpi.h>
#endif

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

struct Spec
{
  Spec(std::string const &backends, int const n_values, int const n_queries,
       int const n_neighbors, int const sort_predicates_int,
       int const buffer_size, PointCloudType const source_point_cloud_type,
       PointCloudType const target_point_cloud_type)
      : backends(backends)
      , n_values(n_values)
      , n_queries(n_queries)
      , n_neighbors(n_neighbors)
      , sort_predicates_int(sort_predicates_int)
      , buffer_size(buffer_size)
      , source_point_cloud_type(source_point_cloud_type)
      , target_point_cloud_type(target_point_cloud_type)
  {
  }

  Spec(std::string const &spec_string)
  {
    std::istringstream ss(spec_string);
    std::string token;

    // clang-format off
    getline(ss, token, '/');  backends = token;
    getline(ss, token, '/');  n_values = std::stoi(token);
    getline(ss, token, '/');  n_queries = std::stoi(token);
    getline(ss, token, '/');  n_neighbors = std::stoi(token);
    getline(ss, token, '/');  sort_predicates_int = std::stoi(token);
    getline(ss, token, '/');  buffer_size = std::stoi(token);
    getline(ss, token, '/');  source_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    getline(ss, token, '/');  target_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    // clang-format on

    if (!(backends == "all" || backends == "serial" || backends == "openmp" ||
          backends == "threads" || backends == "cuda" || backends == "rtree"))
      throw std::runtime_error("Backend " + backends + " invalid!");
  }

  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  int sort_predicates_int;
  int buffer_size;
  PointCloudType source_point_cloud_type;
  PointCloudType target_point_cloud_type;
};

template <typename DeviceType>
Kokkos::View<ArborX::Point *, DeviceType>
constructPoints(int n_values, PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_values);
  // Generate random points uniformly distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  generatePointCloud(point_cloud_type, a, random_points);

  return random_points;
}

template <typename DeviceType>
Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
makeNearestQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_knn_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) =
            ArborX::nearest<ArborX::Point>(random_points(i), n_neighbors);
      });
  return queries;
}

template <typename DeviceType>
Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries(Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  // Radius is computed so that the number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors.
  // Calculation: n_values*(4/3*M_PI*r^3)/(2a)^3 = n_neighbors
  double const r = std::cbrt(static_cast<double>(n_neighbors) * 6. / M_PI);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_radius_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::intersects(ArborX::Sphere{random_points(i), r});
      });
  return queries;
}

template <class TreeType>
void BM_construction(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;
  auto const points =
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type);

  for (auto _ : state)
  {
    auto const start = std::chrono::high_resolution_clock::now();
    TreeType index(points);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_knn_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates_int));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_radius_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(spec.sort_predicates_int)
                    .setBufferSize(spec.buffer_size));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

class KokkosScopeGuard
{
public:
  KokkosScopeGuard(int &argc, char *argv[]) { Kokkos::initialize(argc, argv); }
  ~KokkosScopeGuard() { Kokkos::finalize(); }
};

template <typename TreeType>
void register_benchmark(std::string const &description, Spec const &spec)
{
  auto label_construction = [&](std::string const &tree_name) -> std::string {
    std::string s = std::string("BM_construction<") + tree_name + ">";
    for (auto const &var : {spec.n_values, (int)spec.source_point_cloud_type})
      s += "/" + std::to_string(var);
    return s;
  };
  auto label_knn_search = [&](std::string const &tree_name) -> std::string {
    std::string s = std::string("BM_knn_search<") + tree_name + ">";
    for (auto const &var :
         {spec.n_values, spec.n_queries, spec.n_neighbors,
          spec.sort_predicates_int, (int)spec.source_point_cloud_type,
          (int)spec.target_point_cloud_type})
      s += "/" + std::to_string(var);
    return s;
  };
  auto label_radius_search = [&](std::string const &tree_name) -> std::string {
    std::string s = std::string("BM_radius_search<") + tree_name + ">";
    for (auto const &var :
         {spec.n_values, spec.n_queries, spec.n_neighbors,
          spec.sort_predicates_int, spec.buffer_size,
          (int)spec.source_point_cloud_type, (int)spec.target_point_cloud_type})
      s += "/" + std::to_string(var);
    return s;
  };

  benchmark::RegisterBenchmark(
      label_construction(description).c_str(),
      [=](benchmark::State &state) { BM_construction<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark(
      label_knn_search(description).c_str(),
      [=](benchmark::State &state) { BM_knn_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark(
      label_radius_search(description).c_str(),
      [=](benchmark::State &state) { BM_radius_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

// NOTE Motivation for this class that stores the argument count and values is
// I could not figure out how to make the parser consume arguments with
// Boost.Program_options
// Benchmark removes its own arguments from the command line arguments. This
// means, that by virtue of returning references to internal data members in
// argc() and argv() function, it will necessarily modify the members. It will
// decrease _argc, and "reduce" _argv data. Hence, we must keep a copy of _argv
// that is not modified from the outside to release memory in the destructor
// correctly.
class CmdLineArgs
{
private:
  int _argc;
  std::vector<char *> _argv;
  std::vector<char *> _owner_ptrs;

public:
  CmdLineArgs(std::vector<std::string> const &args, char const *exe)
      : _argc(args.size() + 1)
      , _owner_ptrs{new char[std::strlen(exe) + 1]}
  {
    std::strcpy(_owner_ptrs[0], exe);
    _owner_ptrs.reserve(_argc);
    for (auto const &s : args)
    {
      _owner_ptrs.push_back(new char[s.size() + 1]);
      std::strcpy(_owner_ptrs.back(), s.c_str());
    }
    _argv = _owner_ptrs;
  }

  ~CmdLineArgs()
  {
    for (auto *p : _owner_ptrs)
    {
      delete[] p;
    }
  }

  int &argc() { return _argc; }

  char **argv() { return _argv.data(); }
};

int main(int argc, char *argv[])
{
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  int buffer_size;
  bool sort_predicates;
  std::string source_pt_cloud;
  std::string target_pt_cloud;
  std::vector<std::string> exact_specs;
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&n_values)->default_value(50000), "number of indexable values (source)" )
        ( "queries", bpo::value<int>(&n_queries)->default_value(20000), "number of queries (target)" )
        ( "predicate-sort", bpo::value<bool>(&sort_predicates)->default_value(true), "sort predicates" )
        ( "neighbors", bpo::value<int>(&n_neighbors)->default_value(10), "desired number of results per query" )
        ( "buffer", bpo::value<int>(&buffer_size)->default_value(0), "size for buffer optimization in radius search" )
        ( "source-point-cloud-type", bpo::value<std::string>(&source_pt_cloud)->default_value("filled_box"), "shape of the source point cloud"  )
        ( "target-point-cloud-type", bpo::value<std::string>(&target_pt_cloud)->default_value("filled_box"), "shape of the target point cloud"  )
        ( "no-header", bpo::bool_switch(), "do not print version and hash" )
        ( "exact-spec", bpo::value<std::vector<std::string>>(&exact_specs)->multitoken(), "exact specification (can be specified multiple times for batch)" )
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                   .options(desc)
                                   .allow_unregistered()
                                   .run();
  bpo::store(parsed, vm);
  CmdLineArgs pass_further{
      bpo::collect_unrecognized(parsed.options, bpo::include_positional),
      argv[0]};
  bpo::notify(vm);

  int sort_predicates_int = (sort_predicates ? 1 : 0);

  if (!vm["no-header"].as<bool>())
  {
    std::cout << "ArborX version: " << ArborX::version() << std::endl;
    std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
  }

  if (vm.count("help") > 0)
  {
    // Full list of options consists of Kokkos + Boost.Program_options +
    // Google Benchmark and we still need to call benchmark::Initialize() to
    // get those printed to the standard output.
    std::cout << desc << "\n";
    int ac = 2;
    char *av[] = {(char *)"ignored", (char *)"--help"};
    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit(Kokkos::finalize);
    benchmark::Initialize(&ac, av);
    return 1;
  }

  if (vm.count("exact-spec") > 0)
  {
    for (std::string option :
         {"values", "queries", "predicate-sort", "neighbors", "buffer",
          "source-point-cloud-type", "target-point-cloud-type"})
    {
      if (!vm[option].defaulted())
      {
        std::cout << "Conflicting options: 'exact-spec' and '" << option
                  << "', exiting..." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  benchmark::Initialize(&pass_further.argc(), pass_further.argv());
  // Throw if some of the arguments have not been recognized.
  std::ignore =
      bpo::command_line_parser(pass_further.argc(), pass_further.argv())
          .options(bpo::options_description(""))
          .run();

  std::vector<Spec> specs;
  for (auto const &spec_string : exact_specs)
    specs.emplace_back(spec_string);

  if (vm.count("exact-spec") == 0)
  {
    specs.emplace_back(backends, n_values, n_queries, n_neighbors,
                       sort_predicates_int, buffer_size,
                       to_point_cloud_enum(source_pt_cloud),
                       to_point_cloud_enum(target_pt_cloud));
  }

  for (auto const &spec : specs)
  {
#ifdef KOKKOS_ENABLE_SERIAL
    if (backends == "all" || backends == "serial")
      register_benchmark<ArborX::BVH<Kokkos::Serial::device_type>>(
          "ArborX::BVH<Serial>", spec);
#else
    if (backends == "serial")
      throw std::runtime_error("Serial backend not available!");
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    if (backends == "all" || backends == "openmp")
      register_benchmark<ArborX::BVH<Kokkos::OpenMP::device_type>>(
          "ArborX::BVH<OpenMP>", spec);
#else
    if (backends == "openmp")
      throw std::runtime_error("OpenMP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_THREADS
    if (backends == "all" || backends == "threads")
      register_benchmark<ArborX::BVH<Kokkos::Threads::device_type>>(
          "ArborX::BVH<Threads>", spec);
#else
    if (backends == "threads")
      throw std::runtime_error("Threads backend not available!");
#endif

#ifdef KOKKOS_ENABLE_CUDA
    if (backends == "all" || backends == "cuda")
      register_benchmark<ArborX::BVH<Kokkos::Cuda::device_type>>(
          "ArborX::BVH<Cuda>", spec);
#else
    if (backends == "cuda")
      throw std::runtime_error("CUDA backend not available!");
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    if (backends == "all" || backends == "rtree")
    {
      using BoostRTree = BoostExt::RTree<ArborX::Point>;
      register_benchmark<BoostRTree>("BoostRTree", spec);
    }
#endif
  }

  benchmark::RunSpecifiedBenchmarks();

  Kokkos::finalize();
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
