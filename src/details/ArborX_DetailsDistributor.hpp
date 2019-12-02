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
#ifndef ARBORX_DETAILS_DISTRIBUTOR_HPP
#define ARBORX_DETAILS_DISTRIBUTOR_HPP

#include <ArborX_Config.hpp>

#include <ArborX_DetailsUtils.hpp> // max
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp> // FIXME

#include <algorithm> // max_element
#include <numeric>   // iota
#include <sstream>
#include <vector>

#include <mpi.h>

namespace ArborX
{
namespace Details
{

struct Result
{
  int batch_count;
  int total_count;

  //Result &operator=(const Result &other_result) = default;

  /*Result &operator+=(const Result &other_result)
      {
        batch_count += other_result.batch_count;
        total_count += other_result.total_count;
        return *this;
      }*/

  KOKKOS_INLINE_FUNCTION
  Result volatile &operator+=(const volatile Result &other_result) volatile
  {
    batch_count += other_result.batch_count;
    total_count += other_result.total_count;
    return *this;
  }
};

// Computes the array of indices that sort the input array (in reverse order)
// but also returns the sorted unique elements in that array with the
// corresponding element counts and displacement (offsets)
template <typename InputView, typename OutputView>
static void sortAndDetermineBufferLayout(InputView batched_ranks,
                                         InputView batched_offsets,
                                         OutputView permutation_indices,
                                         std::vector<int> &unique_ranks,
                                         std::vector<int> &counts,
                                         std::vector<int> &offsets)
{
  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(counts.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) ==
                lastElement(batched_offsets));
  ARBORX_ASSERT(batched_ranks.size() + 1 == batched_offsets.size());
  static_assert(
      std::is_same<typename InputView::non_const_value_type, int>::value, "");
  static_assert(std::is_same<typename OutputView::value_type, int>::value, "");

  offsets.push_back(0);

  auto const n = batched_ranks.size();
  if (n == 0 || lastElement(batched_offsets) == 0)
    return;

  // this implements a "sort" which is O(N * R) where (R) is the total number of
  // unique destination ranks. it performs better than other algorithms in the
  // case when (R) is small, but results may vary
  using DeviceType = typename InputView::traits::device_type;
  using ExecutionSpace = typename InputView::traits::execution_space;

  std::cout << "batched_ranks" << std::endl;
  for (unsigned int i=0; i<batched_ranks.size(); ++i)
  {
    std::cout << batched_ranks(i) <<  " ";
  }
  std::cout << std::endl;

  std::cout << "batched_offsets" << std::endl;
  for (unsigned int i=0; i<batched_offsets.size(); ++i)
  {
    std::cout << batched_offsets(i) <<  " ";
  }
  std::cout << std::endl;

  Kokkos::View<int *, DeviceType> device_batched_ranks_duplicate(
      Kokkos::ViewAllocateWithoutInitializing(batched_ranks.label()),
      batched_ranks.size());
  Kokkos::deep_copy(device_batched_ranks_duplicate, batched_ranks);
  Kokkos::View<int *, DeviceType> device_batched_permutation_indices(
      Kokkos::ViewAllocateWithoutInitializing("batched_permutation_indices"),
      batched_ranks.size());

  Kokkos::View<int *, DeviceType> batched_counts("batched_counts",
                                                 batched_offsets.size());
  Kokkos::parallel_for(
      "compute_batch_counts",
      Kokkos::RangePolicy<ExecutionSpace>(0, batched_offsets.size()),
      KOKKOS_LAMBDA(int i) {
        batched_counts[i] = batched_offsets[i + 1] - batched_offsets[i];
      });

  std::cout << "batched_counts" << std::endl;
  for (unsigned int i=0; i<batched_counts.size(); ++i)
  {
    std::cout << batched_counts(i) <<  " ";
  }
  std::cout << std::endl;

  int batch_offset = 0;
  int total_offset = 0;
  while (true)
  {
    int const largest_rank = ArborX::max(device_batched_ranks_duplicate);
    if (largest_rank == -1)
      break;
    Result result = {};

    std::cout << "Before parallel_scan" << std::endl;

    Kokkos::parallel_scan(ARBORX_MARK_REGION("process_biggest_rank_items"),
                          Kokkos::RangePolicy<ExecutionSpace>(0, n),
                          KOKKOS_LAMBDA(int i, Result &update, bool last_pass) {
                            //printf("Here1 %d %lu %d\n", i, n, 0);
                            bool const is_largest_rank =
                                (device_batched_ranks_duplicate(i) ==
                                 largest_rank);
                            //printf("Here2\n");
                            if (is_largest_rank)
                            {
                              if (last_pass)
                              {
                                //printf("Here3\n");
                                device_batched_permutation_indices(i) =
                                    update.batch_count + batch_offset;
                                //printf("Here4\n");
                                device_batched_ranks_duplicate(i) = -1;
                                //printf("Here5\n");
                              }
                              ++update.batch_count;
                              //printf("Here6\n");
                              update.total_count += batched_counts(i);
                              //printf("Here7\n");
                            }
			    //printf("Here7a\n");
                          },
                          result);
    //printf("Here8\n");
    batch_offset += result.batch_count;
    if (result.total_count > 0)
    {
      total_offset += result.total_count;
      unique_ranks.push_back(largest_rank);
      offsets.push_back(total_offset);
    }
  }

  std::cout << "device_batched_permutation_indices" << std::endl;
  for (unsigned int i=0; i<device_batched_permutation_indices.size(); ++i)
  {
    std::cout << device_batched_permutation_indices(i) <<  " ";
  }
  std::cout << std::endl;

  std::cout << "After parallel_scan" << std::endl;

  Kokkos::View<int *, DeviceType> device_batched_permutation_indices_inverse(
      Kokkos::ViewAllocateWithoutInitializing(
          "batched_permutation_indices_inverse"),
      batched_ranks.size());
  Kokkos::parallel_for(
      "invert_batched_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(0, batched_ranks.size()),
      KOKKOS_LAMBDA(int i) {
        device_batched_permutation_indices_inverse(
            device_batched_permutation_indices(i)) = i;
      });

  std::cout << "device_batched_permutation_indices_inverse" << std::endl;
  for (unsigned int i=0; i<device_batched_permutation_indices_inverse.size(); ++i)
  {
    std::cout << device_batched_permutation_indices_inverse(i) <<  " ";
  }
  std::cout << std::endl;

  std::cout << "Before exclusive sum batched offsets" << std::endl;

  InputView exclusive_sum_batched_offsets(
      Kokkos::ViewAllocateWithoutInitializing("exclusive_sum_batched_offsets"),
      batched_offsets.size());
  InputView reordered_batched_counts(
      Kokkos::ViewAllocateWithoutInitializing("reordered_batched_counts"),
      batched_offsets.size());
  std::cout << "reordered_batches " << batched_offsets.size() << std::endl;
  Kokkos::parallel_for(
      "iota",
      Kokkos::RangePolicy<ExecutionSpace>(0, batched_offsets.size() - 1),
      KOKKOS_LAMBDA(int j) {
      //printf("reordered_batched_counts(%d) = batched_counts(%d)\n", j, device_batched_permutation_indices_inverse(j));
      //printf("reordered_batched_counts(%d) = %d\n", j, batched_counts(device_batched_permutation_indices_inverse(j)));
        reordered_batched_counts(j) =
            batched_counts(device_batched_permutation_indices_inverse(j));
      });

  std::cout << "reordered_batched_counts" << std::endl;
  for (unsigned int i=0; i<reordered_batched_counts.size(); ++i)
  {
    std::cout << reordered_batched_counts(i) <<  " ";
  }
  std::cout << std::endl;

  ArborX::exclusivePrefixSum(batched_counts, exclusive_sum_batched_offsets);
  std::cout << "permutation_inverse" << std::endl;
  Kokkos::View<int *, DeviceType> device_permutation_indices_inverse(
      Kokkos::ViewAllocateWithoutInitializing(
          "device_permutation_indices_inverse"),
      permutation_indices.size());

    std::cout << "exclusive_sum_batched_offsets" << std::endl;
  for (unsigned int i=0; i<exclusive_sum_batched_offsets.size(); ++i)
  {
    std::cout << exclusive_sum_batched_offsets(i) <<  " ";
  }
  std::cout << std::endl;

  const auto batched_counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batched_counts);
  const auto batched_permutation_indices_inverse =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          device_batched_permutation_indices);

  std::cout << "iota " << std::endl;

  int starting_permutation = 0;
  for (unsigned int i = 0; i < device_batched_permutation_indices.size(); ++i)
  {
    int n_batch_entries =
        reordered_batched_counts[i];
    std::cout << exclusive_sum_batched_offsets(
                  device_batched_permutation_indices_inverse(i))<< "+" << n_batch_entries << " <= " << device_permutation_indices_inverse.size() << std::endl;
    assert(exclusive_sum_batched_offsets(
                  device_batched_permutation_indices_inverse(i))+n_batch_entries<=device_permutation_indices_inverse.size());
    Kokkos::parallel_for(
        "iota", Kokkos::RangePolicy<ExecutionSpace>(0, n_batch_entries),
        KOKKOS_LAMBDA(int j) {
          device_permutation_indices_inverse(starting_permutation + j) =
              exclusive_sum_batched_offsets(
                  device_batched_permutation_indices_inverse(i)) +
              j;
        });
    starting_permutation += n_batch_entries;
  }

   std::cout << "device_permutation_indices_inverse" << std::endl;
  for (unsigned int i=0; i<device_permutation_indices_inverse.size(); ++i)
  {
    std::cout << device_permutation_indices_inverse(i) <<  " ";
  }
  std::cout << std::endl;

  std::cout << "invert permutation" << std::endl;

  Kokkos::View<int *, DeviceType> device_permutation_indices(
      Kokkos::ViewAllocateWithoutInitializing("device_permutation_indices"),
      permutation_indices.size());
  Kokkos::parallel_for(
      "invert_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(0, permutation_indices.size()),
      KOKKOS_LAMBDA(int i) {
        printf("device_permutation_indices(%d) = %d < %lu\n", device_permutation_indices_inverse(i), i, permutation_indices.size());
	assert(device_permutation_indices_inverse(i)<permutation_indices.size());
        device_permutation_indices(device_permutation_indices_inverse(i)) = i;
      });

     std::cout << "device_permutation_indices" << std::endl;
  for (unsigned int i=0; i<device_permutation_indices.size(); ++i)
  {
    std::cout << device_permutation_indices(i) <<  " ";
  }
  std::cout << std::endl;

  std::cout << "end" << std::endl;

  counts.reserve(offsets.size() - 1);
  for (unsigned int i = 1; i < offsets.size(); ++i)
    counts.push_back(offsets[i] - offsets[i - 1]);
  Kokkos::deep_copy(permutation_indices, device_permutation_indices);
  //  ARBORX_ASSERT(offsets.back() == static_cast<int>(ranks.size()));
  ARBORX_ASSERT(unique_ranks.size() == counts.size());
  ARBORX_ASSERT(offsets.size() == unique_ranks.size() + 1);
}

// Computes the array of indices that sort the input array (in reverse order)
// but also returns the sorted unique elements in that array with the
// corresponding element counts and displacement (offsets)
template <typename InputView, typename OutputView>
static void sortAndDetermineBufferLayout(InputView ranks,
                                         OutputView permutation_indices,
                                         std::vector<int> &unique_ranks,
                                         std::vector<int> &counts,
                                         std::vector<int> &offsets)
{
  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(counts.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) == ranks.extent_int(0));
  static_assert(
      std::is_same<typename InputView::non_const_value_type, int>::value, "");
  static_assert(std::is_same<typename OutputView::value_type, int>::value, "");

  offsets.push_back(0);

  auto const n = ranks.extent_int(0);
  if (n == 0)
    return;

  // this implements a "sort" which is O(N * R) where (R) is the total number of
  // unique destination ranks. it performs better than other algorithms in the
  // case when (R) is small, but results may vary
  using DeviceType = typename InputView::traits::device_type;
  using ExecutionSpace = typename InputView::traits::execution_space;

  Kokkos::View<int *, DeviceType> device_ranks_duplicate(
      Kokkos::ViewAllocateWithoutInitializing(ranks.label()), ranks.size());
  Kokkos::deep_copy(device_ranks_duplicate, ranks);
  auto device_permutation_indices =
      Kokkos::create_mirror_view(DeviceType(), permutation_indices);
  int offset = 0;
  while (true)
  {
    int const largest_rank = ArborX::max(device_ranks_duplicate);
    if (largest_rank == -1)
      break;
    unique_ranks.push_back(largest_rank);
    int result = 0;
    Kokkos::parallel_scan(ARBORX_MARK_REGION("process_biggest_rank_items"),
                          Kokkos::RangePolicy<ExecutionSpace>(0, n),
                          KOKKOS_LAMBDA(int i, int &update, bool last_pass) {
                            bool const is_largest_rank =
                                (device_ranks_duplicate(i) == largest_rank);
                            if (is_largest_rank)
                            {
                              if (last_pass)
                              {
                                device_permutation_indices(i) = update + offset;
                                device_ranks_duplicate(i) = -1;
                              }
                              ++update;
                            }
                          },
                          result);
    offset += result;
    offsets.push_back(offset);
  }
  counts.reserve(offsets.size() - 1);
  for (unsigned int i = 1; i < offsets.size(); ++i)
    counts.push_back(offsets[i] - offsets[i - 1]);
  Kokkos::deep_copy(permutation_indices, device_permutation_indices);
  ARBORX_ASSERT(offsets.back() == static_cast<int>(ranks.size()));
}

template <typename DeviceType>
class Distributor
{
public:
  Distributor(MPI_Comm comm)
      : _comm(comm)
      , _permute{Kokkos::ViewAllocateWithoutInitializing("permute"), 0}
  {
  }

  template <typename View>
  size_t createFromSends(View const &batched_destination_ranks,
                         View const &batch_offsets)
  {
    static_assert(View::rank == 1, "");
    static_assert(std::is_same<typename View::non_const_value_type, int>::value,
                  "");
    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);

    reallocWithoutInitializing(_permute, lastElement(batch_offsets));
    sortAndDetermineBufferLayout(batched_destination_ranks, batch_offsets,
                                 _permute, _destinations, _dest_counts,
                                 _dest_offsets);

    std::vector<int> src_counts_dense(comm_size);
    int const dest_size = _destinations.size();
    for (int i = 0; i < dest_size; ++i)
    {
      src_counts_dense[_destinations[i]] = _dest_counts[i];
    }
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, src_counts_dense.data(), 1,
                 MPI_INT, _comm);

    _src_offsets.push_back(0);
    for (int i = 0; i < comm_size; ++i)
      if (src_counts_dense[i] > 0)
      {
        _sources.push_back(i);
        _src_counts.push_back(src_counts_dense[i]);
        _src_offsets.push_back(_src_offsets.back() + _src_counts.back());
      }

    return _src_offsets.back();
  }

  template <typename View>
  size_t createFromSends(View const &destination_ranks)
  {
    static_assert(View::rank == 1, "");
    static_assert(std::is_same<typename View::non_const_value_type, int>::value,
                  "");
    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);

    reallocWithoutInitializing(_permute, destination_ranks.size());
    sortAndDetermineBufferLayout(destination_ranks, _permute, _destinations,
                                 _dest_counts, _dest_offsets);

    std::vector<int> src_counts_dense(comm_size);
    int const dest_size = _destinations.size();
    for (int i = 0; i < dest_size; ++i)
    {
      src_counts_dense[_destinations[i]] = _dest_counts[i];
    }
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, src_counts_dense.data(), 1,
                 MPI_INT, _comm);

    _src_offsets.push_back(0);
    for (int i = 0; i < comm_size; ++i)
      if (src_counts_dense[i] > 0)
      {
        _sources.push_back(i);
        _src_counts.push_back(src_counts_dense[i]);
        _src_offsets.push_back(_src_offsets.back() + _src_counts.back());
      }

    return _src_offsets.back();
  }

  template <typename View>
  void doPostsAndWaits(typename View::const_type const &exports,
                       size_t num_packets, View const &imports) const
  {
    ARBORX_ASSERT(num_packets * _src_offsets.back() == imports.size());
    ARBORX_ASSERT(num_packets * _dest_offsets.back() == exports.size());

    using ValueType = typename View::value_type;
    using ExecutionSpace = typename View::execution_space;
    static_assert(View::rank == 1, "");

    static_assert(
        std::is_same<typename View::memory_space,
                     typename decltype(_permute)::memory_space>::value,
        "");
#ifndef ARBORX_USE_CUDA_AWARE_MPI
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "");
#endif

    Kokkos::View<ValueType *, typename View::traits::device_type> dest_buffer(
        Kokkos::ViewAllocateWithoutInitializing("destination_buffer"),
        exports.size());

    // We need to create a local copy to avoid capturing a member variable
    // (via the 'this' pointer) which we can't do using a KOKKOS_LAMBDA.
    // Use KOKKOS_CLASS_LAMBDA when we require C++17.
    auto const permute_copy = _permute;

    Kokkos::parallel_for("copy_destinations_permuted",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             0, _dest_offsets.back() * num_packets),
                         KOKKOS_LAMBDA(int const k) {
                           int const i = k / num_packets;
                           int const j = k % num_packets;
			   printf("dest_buffer(%lu) = exports[%lu]\n", num_packets * permute_copy[i] + j, num_packets * i + j);
                           dest_buffer(num_packets * permute_copy[i] + j) =
                               exports[num_packets * i + j];
                         });

    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);
    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    std::vector<MPI_Request> requests;
    requests.reserve(outdegrees + indegrees);
    for (int i = 0; i < indegrees; ++i)
    {
      if (_sources[i] != comm_rank)
      {
        auto const message_size =
            _src_counts[i] * num_packets * sizeof(ValueType);
        auto const receive_buffer_ptr =
            imports.data() + _src_offsets[i] * num_packets;
        requests.emplace_back();
        MPI_Irecv(receive_buffer_ptr, message_size, MPI_BYTE, _sources[i], 123,
                  _comm, &requests.back());
      }
    }

    // make sure the data in dest_buffer has been copied before sending it.
    ExecutionSpace().fence();

    for (int i = 0; i < outdegrees; ++i)
    {
      auto const message_size =
          _dest_counts[i] * num_packets * sizeof(ValueType);
      auto const send_buffer_ptr =
          dest_buffer.data() + _dest_offsets[i] * num_packets;
      if (_destinations[i] == comm_rank)
      {
        auto const it = std::find(_sources.begin(), _sources.end(), comm_rank);
        ARBORX_ASSERT(it != _sources.end());
        auto const position = it - _sources.begin();
        auto const receive_buffer_ptr =
            imports.data() + _src_offsets[position] * num_packets;

        Kokkos::View<ValueType *, typename View::traits::device_type,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            receive_view(receive_buffer_ptr, message_size / sizeof(ValueType));
        Kokkos::View<const ValueType *, typename View::traits::device_type,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            send_view(send_buffer_ptr, message_size / sizeof(ValueType));
        Kokkos::deep_copy(receive_view, send_view);
      }
      else
      {
        requests.emplace_back();
        MPI_Isend(send_buffer_ptr, message_size, MPI_BYTE, _destinations[i],
                  123, _comm, &requests.back());
      }
    }
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
  size_t getTotalReceiveLength() const { return _src_offsets.back(); }
  size_t getTotalSendLength() const { return _dest_offsets.back(); }

private:
  MPI_Comm _comm;
#ifdef ARBORX_USE_CUDA_AWARE_MPI
  Kokkos::View<int *, DeviceType> _permute;
#else
  Kokkos::View<int *, Kokkos::HostSpace> _permute;
#endif
  std::vector<int> _dest_offsets;
  std::vector<int> _dest_counts;
  std::vector<int> _src_offsets;
  std::vector<int> _src_counts;
  std::vector<int> _sources;
  std::vector<int> _destinations;
};

} // namespace Details
} // namespace ArborX

#endif
