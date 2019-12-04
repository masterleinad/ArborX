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

template <typename InputView, typename OutputView>
void invertPermutation(const InputView &input, OutputView &output)
{
  static_assert(std::is_integral<typename InputView::value_type>::value, "");
  static_assert(std::is_integral<typename OutputView::value_type>::value, "");
  static_assert(InputView::rank == 1, "");
  static_assert(OutputView::rank == 1, "");
  ARBORX_ASSERT(input.size() == output.size());

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("invert_permutation"),
      Kokkos::RangePolicy<typename InputView::execution_space>(0, input.size()),
      KOKKOS_LAMBDA(std::size_t i) { output(input(i)) = i; });
}

// Used in the batched version of sortAndDetermineBufferLayout
struct Result
{
  int batch_count;
  int total_count;

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
// This overload uses batched ranks and offsets.
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

  using DeviceType = typename InputView::traits::device_type;
  using ExecutionSpace = typename InputView::traits::execution_space;

  Kokkos::View<int *, DeviceType> device_batched_ranks_duplicate(
      Kokkos::ViewAllocateWithoutInitializing(batched_ranks.label()),
      batched_ranks.size());
  Kokkos::deep_copy(device_batched_ranks_duplicate, batched_ranks);
  Kokkos::View<int *, DeviceType> device_batched_permutation_indices(
      Kokkos::ViewAllocateWithoutInitializing("batched_permutation_indices"),
      batched_ranks.size());

  Kokkos::View<int *, DeviceType> batched_counts("batched_counts",
                                                 batched_offsets.size() - 1);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("compute_batch_counts"),
      Kokkos::RangePolicy<ExecutionSpace>(0, batched_offsets.size() - 1),
      KOKKOS_LAMBDA(int i) {
        batched_counts[i] = batched_offsets[i + 1] - batched_offsets[i];
      });

  // step 1: compute batch permutation, total unique_ranks, total offsets
  //         and total counts
  int batch_offset = 0;
  int total_offset = 0;
  while (true)
  {
    int const largest_rank = ArborX::max(device_batched_ranks_duplicate);
    if (largest_rank == -1)
      break;
    Result result = {};

    Kokkos::parallel_scan(ARBORX_MARK_REGION("process_biggest_rank_items"),
                          Kokkos::RangePolicy<ExecutionSpace>(0, n),
                          KOKKOS_LAMBDA(int i, Result &update, bool last_pass) {
                            bool const is_largest_rank =
                                (device_batched_ranks_duplicate(i) ==
                                 largest_rank);
                            if (is_largest_rank)
                            {
                              if (last_pass)
                              {
                                device_batched_permutation_indices(i) =
                                    update.batch_count + batch_offset;
                                device_batched_ranks_duplicate(i) = -1;
                              }
                              ++update.batch_count;
                              update.total_count += batched_counts(i);
                            }
                          },
                          result);
    batch_offset += result.batch_count;
    if (result.total_count > 0)
    {
      total_offset += result.total_count;
      unique_ranks.push_back(largest_rank);
      offsets.push_back(total_offset);
    }
  }
  counts.reserve(offsets.size() - 1);
  for (unsigned int i = 1; i < offsets.size(); ++i)
    counts.push_back(offsets[i] - offsets[i - 1]);
  ARBORX_ASSERT(unique_ranks.size() == counts.size());
  ARBORX_ASSERT(offsets.size() == unique_ranks.size() + 1);

  // step 2: prepare for creating the total permutation by reordering batch
  //         counts and computing the new exclusive cumulative sum for these
  Kokkos::View<int *, DeviceType> device_batched_permutation_indices_inverse(
      Kokkos::ViewAllocateWithoutInitializing(
          "batched_permutation_indices_inverse"),
      batched_ranks.size());
  ArborX::Details::invertPermutation(
      device_batched_permutation_indices,
      device_batched_permutation_indices_inverse);

  InputView reordered_batched_counts(
      Kokkos::ViewAllocateWithoutInitializing("reordered_batched_counts"),
      batched_offsets.size()); // the last entry is unused
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("reorder_batch_counts"),
      Kokkos::RangePolicy<ExecutionSpace>(0, batched_offsets.size() - 1),
      KOKKOS_LAMBDA(int j) {
        reordered_batched_counts(j) =
            batched_counts(device_batched_permutation_indices_inverse(j));
      });

  InputView exclusive_sum_reordered_batched_offsets(
      Kokkos::ViewAllocateWithoutInitializing(
          "exclusive_sum_reordered_batched_offsets"),
      batched_offsets.size());
  ArborX::exclusivePrefixSum(reordered_batched_counts,
                             exclusive_sum_reordered_batched_offsets);

  // step 3: compute the total permutation
  Kokkos::View<int *, DeviceType> device_permutation_indices_inverse(
      Kokkos::ViewAllocateWithoutInitializing(
          "device_permutation_indices_inverse"),
      permutation_indices.size());

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("set_permutation_indices"),
      Kokkos::RangePolicy<ExecutionSpace>(
          0, device_batched_permutation_indices.size()),
      KOKKOS_LAMBDA(int i) {
        int n_batch_entries = reordered_batched_counts[i];
        for (int j = 0; j < n_batch_entries; ++j)
        {
          device_permutation_indices_inverse(
              exclusive_sum_reordered_batched_offsets(i) + j) =
              batched_offsets(device_batched_permutation_indices_inverse(i)) +
              j;
        }
      });

  auto device_permutation_indices =
      Kokkos::create_mirror_view(DeviceType(), permutation_indices);
  ArborX::Details::invertPermutation(device_permutation_indices_inverse,
                                     device_permutation_indices);
  Kokkos::deep_copy(permutation_indices, device_permutation_indices);
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
