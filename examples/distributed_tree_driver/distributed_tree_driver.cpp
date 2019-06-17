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

#include <ArborX_DistributedSearchTree.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath> // cbrt
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <mpi.h>

struct HelpPrinted
{
};

using namespace boost::accumulators;

// Poor man's replacement for Teuchos::TimeMonitor
class TimeMonitor
{
  class Timer
  {
    using data_type =
        accumulator_set<double, stats<tag::min, tag::max, tag::count, tag::mean,
                                      tag::median, tag::variance>>;
    data_type _statistics;
    bool _started = false;
    const MPI_Comm _mpi_comm;
    std::chrono::high_resolution_clock::time_point _tick;

  public:
    Timer(const MPI_Comm mpi_communicator = MPI_COMM_WORLD) : _mpi_comm(mpi_communicator) {}

    // Prevent accidental copy construction
    Timer(Timer const &) = delete;

    // Reset the lap timer to start measuring a new iteration.
    void start()
    {
      assert(!_started);
      _tick = std::chrono::high_resolution_clock::now();
      _started = true;
    }

    // Take the maximum of the elapsed lap time across all the MPI processes and store it in the boost::accumulator object.
    void stop()
    {
      assert(_started);
      std::chrono::duration<double> duration =
          std::chrono::high_resolution_clock::now() - _tick;
      MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_DOUBLE, MPI_MAX, _mpi_comm);
      _statistics(duration.count());
      _started = false;
    }

    data_type const &get_statistics() const { return _statistics; }

    // Following CppCon 2015: Bryce Adelstein-Lelbach “Benchmarking C++ Code" (https://www.youtube.com/watch?v=zWxSZcpeS8Q,
    // https://github.com/CppCon/CppCon2015/tree/master/Presentations/Benchmarking%20C%2B%2B%20Code)
    // given a relative margin of error e_m, a quantile z corresponding to the given confidence to be achieved,
    // the (estimated) mean \nu and the (estimated) standard deviation \sigma, the number of required samples n can be computed as
    // n = ((z \sigma)/(e_m/2 \mu))^2.
    int estimate_required_sample_size(double confidence,
                                      double relative_error_margin) const
    {
      int const n_measurements = count(_statistics);
      boost::math::students_t const dist(n_measurements - 1);
      double const z =
          boost::math::quantile(complement(dist, (1 - confidence) / 2));

      double const current_stddev = std::sqrt(
          variance(_statistics) * (n_measurements / (n_measurements - 1.)));
      double const current_mean = mean(_statistics);
      return static_cast<int>(std::ceil(
          z * current_stddev / (relative_error_margin * current_mean / 2.)));
    }
  };

  using container_type = std::map<std::string, Timer>;
  using entry_reference_type = container_type::reference;
  using entry_const_reference_type = container_type::const_reference;
  const MPI_Comm _mpi_comm;
  container_type _data;

public:
  TimeMonitor(const MPI_Comm mpi_communicator = MPI_COMM_WORLD) : _mpi_comm(mpi_communicator) {}

  // Provide access to timers by their names.
  Timer &getTimer(std::string name) { return _data[name]; }

  // Return a non-modifyable reference to std::vector of all timers.
  container_type const &getAllTimer() const { return _data; }

  // Estimate the number of samples required to achieve a given margin of error
  // with a given confidence as the maximum of the estimates for all the timers stored.
  int estimate_required_sample_size(double confidence,
                                    double relative_error_margin) const
  {
    return std::accumulate(
        _data.begin(), _data.end(), 0,
        [confidence, relative_error_margin](int current_max,
                                            entry_const_reference_type entry) {
          return std::max(current_max,
                          entry.second.estimate_required_sample_size(
                              confidence, relative_error_margin));
        });
  }

  // Print statistics about all the timersi stored using os.
  void summarize(std::ostream &os = std::cout)
  {
    // FIXME Haven't tried very hard to format the output.
    int comm_size;
    MPI_Comm_size(_mpi_comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(_mpi_comm, &comm_rank);
    int n_timers = _data.size();
    if (comm_size == 1)
    {
      os << "========================================\n\n";
      os << "TimeMonitor results over 1 processor\n\n";
      os << "Timer Name\tGlobal Time\n";
      os << "----------------------------------------\n";
      for (auto const &timer : _data)
      {
        auto const &statistics = timer.second.get_statistics();
        os << timer.first << "\t" << mean(statistics) << ", "
           << count(statistics) << ", " << mean(statistics) << ", "
           << variance(statistics) << "\n";
      }
      os << "========================================\n";
      return;
    }
    std::vector<double> all_entries(comm_size * n_timers);
    std::cout << "first: " << _data.begin()->first << ", "
              << mean(_data.begin()->second.get_statistics()) << std::endl;
    std::transform(_data.begin(), _data.end(),
                   all_entries.begin() + comm_rank * n_timers,
                   [](entry_const_reference_type x) {
                     return mean(x.second.get_statistics());
                   });
    // FIXME No guarantee that all processors have the same timers!
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_entries.data(),
                  n_timers, MPI_DOUBLE, _mpi_comm);
    if (comm_rank == 0)
    {
      os << "========================================\n\n";
      os << "TimeMonitor results over " << comm_size << " processors\n";
      os << "Timer Name\tMinOverProcs\tMeanOverProcs\tMaxOverProcs\n";
      os << "----------------------------------------\n";
    }
    std::vector<double> tmp(comm_size);
    auto timer_it = _data.begin();
    for (int i = 0; i < n_timers; ++i, ++timer_it)
    {
      for (int j = 0; j < comm_size; ++j)
      {
        tmp[j] = all_entries[j * n_timers + i];
      }
      auto min = *std::min_element(tmp.begin(), tmp.end());
      auto max = *std::max_element(tmp.begin(), tmp.end());
      auto mean = std::accumulate(tmp.begin(), tmp.end(), 0.) / comm_size;
      if (comm_rank == 0)
      {
        os << timer_it->first << "\t" << min << "\t" << mean << "\t" << max
           << "\n";
      }
    }
    if (comm_rank == 0)
    {
      os << "========================================\n";
    }
  }
};

namespace bpo = boost::program_options;

template <class NO>
void main_(std::vector<std::string> const &args, TimeMonitor &time_monitor)
{
  using DeviceType = typename NO::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;

  int n_values;
  int n_queries;
  int n_neighbors;
  double overlap;
  int partition_dim;
  bool perform_knn_search = true;
  bool perform_radius_search = true;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&n_values)->default_value(50000), "number of indexable values (source) per MPI rank" )
        ( "queries", bpo::value<int>(&n_queries)->default_value(20000), "number of queries (target) per MPI rank" )
        ( "neighbors", bpo::value<int>(&n_neighbors)->default_value(10), "desired number of results per query" )
        ( "overlap", bpo::value<double>(&overlap)->default_value(0.), "overlap of the point clouds. 0 means the clouds are built "
                                                                      "next to each other. 1 means that there are built at the "
                                                                      "same place. Negative values and values larger than two "
                                                                      "means that the clouds are separated" )
        ( "partition_dim", bpo::value<int>(&partition_dim)->default_value(3), "number of dimension used by the partitioning of the global "
                                                                              "point cloud. 1 -> local clouds are aligned on a line, 2 -> "
                                                                              "local clouds form a board, 3 -> local clouds form a box" )
        ( "do-not-perform-knn-search", "skip kNN search" )
        ( "do-not-perform-radius-search", "skip radius search" )
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(args).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    throw HelpPrinted();
  }

  if (vm.count("do-not-perform-knn-search"))
    perform_knn_search = false;
  if (vm.count("do-not-perform-radius-search"))
    perform_radius_search = false;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  Kokkos::View<ArborX::Point *, DeviceType> random_points("random_points", 0);
  {
    // Random points are "reused" between building the tree and performing
    // queries. Note that this means that for the points in the middle of
    // the local domains there won't be any communication.
    auto n = std::max(n_values, n_queries);
    Kokkos::resize(random_points, n);

    auto random_points_host = Kokkos::create_mirror_view(random_points);

    // Generate random points uniformely distributed within a box.
    auto const a = std::cbrt(n_values);
    std::uniform_real_distribution<double> distribution(-a, +a);
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
      return distribution(generator);
    };

    double offset_x = 0.;
    double offset_y = 0.;
    double offset_z = 0.;
    // Change the geometry of the problem. In 1D, all the point clouds are
    // aligned on a line. In 2D, the point clouds create a board and in 3D,
    // they create a box.
    switch (partition_dim)
    {
    case 1:
    {
      offset_x = 2. * (1. - overlap) * a * comm_rank;

      break;
    }
    case 2:
    {
      int i_max = std::ceil(std::sqrt(comm_size));
      int i = comm_rank % i_max;
      int j = comm_rank / i_max;
      offset_x = 2. * (1. - overlap) * a * i;
      offset_y = 2. * (1. - overlap) * a * j;

      break;
    }
    case 3:
    {
      int i_max = std::ceil(std::cbrt(comm_size));
      int j_max = i_max;
      int i = comm_rank % i_max;
      int j = (comm_rank / i_max) % j_max;
      int k = comm_rank / (i_max * j_max);
      offset_x = 2. * (1. - overlap) * a * i;
      offset_y = 2. * (1. - overlap) * a * j;
      offset_z = 2. * (1. - overlap) * a * k;

      break;
    }
    default:
    {
      throw std::runtime_error("partition_dim should be 1, 2, or 3");
    }
    }

    for (int i = 0; i < n; ++i)
      random_points_host(i) = {
          {offset_x + random(), offset_y + random(), offset_z + random()}};
    Kokkos::deep_copy(random_points, random_points_host);
  }

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes(
      Kokkos::ViewAllocateWithoutInitializing("bounding_boxes"), n_values);
  Kokkos::parallel_for("bvh_driver:construct_bounding_boxes",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_values),
                       KOKKOS_LAMBDA(int i) {
                         double const x = random_points(i)[0];
                         double const y = random_points(i)[1];
                         double const z = random_points(i)[2];
                         bounding_boxes(i) = {{{x - 1., y - 1., z - 1.}},
                                              {{x + 1., y + 1., z + 1.}}};
                       });
  Kokkos::fence();

  auto &construction = time_monitor.getTimer("construction");
  MPI_Barrier(comm);
  construction.start();
  ArborX::DistributedSearchTree<DeviceType> distributed_tree(comm,
                                                             bounding_boxes);
  construction.stop();

  std::ostream &os = std::cout;
  if (comm_rank == 0)
    os << "contruction done\n";

  if (perform_knn_search)
  {
    Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
        Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
    Kokkos::parallel_for("bvh_driver:setup_knn_search_queries",
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int i) {
                           queries(i) = ArborX::nearest<ArborX::Point>(
                               random_points(i), n_neighbors);
                         });
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    Kokkos::View<int *, DeviceType> ranks("ranks", 0);

    auto &knn = time_monitor.getTimer("knn");
    MPI_Barrier(comm);
    knn.start();
    distributed_tree.query(queries, indices, offset, ranks);
    knn.stop();

    if (comm_rank == 0)
      os << "knn done\n";
  }

  if (perform_radius_search)
  {
    Kokkos::View<ArborX::Within *, DeviceType> queries(
        Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
    // radius chosen in order to control the number of results per query
    // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
    // inserted into the tree (mid-point between half-edge and
    // half-diagonal)
    double const r =
        2. * std::cbrt(static_cast<double>(n_neighbors) * 3. / (4. * M_PI)) -
        (1. + std::sqrt(3.)) / 2.;
    Kokkos::parallel_for("bvh_driver:setup_radius_search_queries",
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int i) {
                           queries(i) = ArborX::within(random_points(i), r);
                         });
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    Kokkos::View<int *, DeviceType> ranks("ranks", 0);

    auto &radius = time_monitor.getTimer("radius");
    MPI_Barrier(comm);
    radius.start();
    distributed_tree.query(queries, indices, offset, ranks);
    radius.stop();

    if (comm_rank == 0)
      os << "radius done\n";
  }
}

template <class NO>
int run(std::vector<std::string> const &args, TimeMonitor &time_monitor)
{
  const unsigned int n_sample = 10;
  for (unsigned int i = 0; i < n_sample; ++i)
    main_<NO>(args, time_monitor);

  // 95% confidence
  boost::math::students_t const dist(n_sample - 1);
  double const z = boost::math::quantile(complement(dist, (1 - 0.95) / 2));
  std::cout << "initial z: " << z << std::endl;

  auto const &statistics =
      time_monitor.getAllTimer().begin()->second.get_statistics();
  double const sample_stddev =
      std::sqrt(variance(statistics) * (n_sample / (n_sample - 1.)));
  double const sample_mean = mean(statistics);
  double const error_margin = sample_mean / 100.;
  auto const n =
      static_cast<int>(std::ceil(z * sample_stddev / (error_margin / 2.)));

  auto const n_new = time_monitor.estimate_required_sample_size(
      /*confidence = */ 0.95, /*relative_error_margin = */ 1. / 100.);

  std::cout << "estimated " << n << " " << n_new << " iterations" << std::endl;

  auto total_n = n_sample + n;
  boost::math::students_t const final_dist(total_n - 1);
  double const final_z =
      boost::math::quantile(complement(final_dist, (1 - 0.95) / 2));

  std::cout << "final z: " << final_z << std::endl;

  for (unsigned int i = 0; i < n; ++i)
    main_<NO>(args, time_monitor);

  double const final_stddev =
      std::sqrt(variance(statistics) * (total_n / (total_n - 1.)));
  double const final_sample_mean = mean(statistics);

  std::cout << "value is between "
            << sample_mean - z * final_stddev / sqrt(total_n) << " and "
            << sample_mean - z * final_stddev / sqrt(total_n) << std::endl;
  std::cout << "min" << min(statistics) << std::endl;
  std::cout << "max" << max(statistics) << std::endl;
  std::cout << "variance" << final_stddev << std::endl;
  std::cout << "sample_variance" << sample_stddev << std::endl;

  time_monitor.summarize();
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  bool success = true;

  try
  {
    std::string node;
    // NOTE Lame trick to get a valid default value
#if defined(KOKKOS_ENABLE_CUDA)
    node = "cuda";
#elif defined(KOKKOS_ENABLE_OPENMP)
    node = "openmp";
#elif defined(KOKKOS_ENABLE_SERIAL)
    node = "serial";
#endif
    bpo::options_description desc("Not a very helpful name");
    // clang-format off
        desc.add_options()
            ( "node", bpo::value<std::string>(&node), "node type (serial | openmp | cuda)" )
        ;
    // clang-format on
    bpo::variables_map vm;
    bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                     .options(desc)
                                     .allow_unregistered()
                                     .run();
    bpo::store(parsed, vm);
    std::vector<std::string> pass_further =
        bpo::collect_unrecognized(parsed.options, bpo::include_positional);

    if (std::find_if(pass_further.begin(), pass_further.end(),
                     [](std::string const &x) { return x == "--help"; }) !=
        pass_further.end())
    {
      std::cout << desc << "\n";
    }

    TimeMonitor time_monitor;

    if (node == "serial")
    {
#ifdef KOKKOS_ENABLE_SERIAL
      typedef Kokkos::Serial Node;
      run<Node>(pass_further, time_monitor);
#else
      throw std::runtime_error("Serial node type is disabled");
#endif
    }
    else if (node == "openmp")
    {
#ifdef KOKKOS_ENABLE_OPENMP
      typedef Kokkos::OpenMP Node;
      run<Node>(pass_further, time_monitor);
#else
      throw std::runtime_error("OpenMP node type is disabled");
#endif
    }
    else if (node == "cuda")
    {
#ifdef KOKKOS_ENABLE_CUDA
      typedef Kokkos::CudaUVMSpace Node;
      run<Node>(pass_further, time_monitor);
#else
      throw std::runtime_error("CUDA node type is disabled");
#endif
    }
    else
    {
      throw std::runtime_error("Unrecognized node type");
    }
  }
  catch (HelpPrinted const &)
  {
    // Do nothing, it was a successful run. Just clean up things below.
  }
  catch (std::exception const &e)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "processor " << rank
              << " caught a std::exception: " << e.what() << "\n";
    success = false;
  }
  catch (...)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "processor " << rank << " caught some kind of exception\n";
    success = false;
  }

  Kokkos::finalize();

  MPI_Finalize();

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
