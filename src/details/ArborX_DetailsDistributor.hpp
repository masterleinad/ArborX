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

#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp> // FIXME

#include <algorithm> // max_element
#include <numeric>   // iota
#include <sstream>
#include <vector>
#include <cassert>

#include <mpi.h>

namespace ArborX
{
namespace Details
{

// NOTE: We were getting a compile error on CUDA when using a KOKKOS_LAMBDA.
template <typename DeviceType>
class BiggestRankItemsFunctor
{
public:
  BiggestRankItemsFunctor(
      const Kokkos::View<int *, DeviceType> &ranks_duplicate,
      const Kokkos::View<int, DeviceType> &largest_rank,
      const Kokkos::View<int *, DeviceType> &permutation_indices,
      const Kokkos::View<int, DeviceType> &offset)
      : _ranks_duplicate(ranks_duplicate)
      , _largest_rank(largest_rank)
      , _permutation_indices(permutation_indices)
      , _offset(offset)
  {
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i, int &update,
                                         bool last_pass) const
  {
    const bool is_largest_rank = (_ranks_duplicate(i) == _largest_rank());
    if (is_largest_rank)
    {
      if (last_pass)
      {
        _permutation_indices(i) = update + _offset();
        _ranks_duplicate(i) = -1;
      }
      ++update;
    }
    if (last_pass && i + 1 == _ranks_duplicate.extent(0))
      _offset() += update;
  }

private:
  const Kokkos::View<int *, DeviceType> _ranks_duplicate;
  const Kokkos::View<int, DeviceType> _largest_rank;
  const Kokkos::View<int *, DeviceType> _permutation_indices;
  const Kokkos::View<int, DeviceType> _offset;
};

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
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<typename OutputView::memory_space,
                                      Kokkos::HostSpace>::accessible,
      "");

  offsets.push_back(0);

  auto const n = ranks.extent_int(0);
  if (n == 0)
    return;

  std::cout << "ranks" << std::endl;
  for (unsigned int i=0; i<ranks.size(); ++i)
    std::cout << ranks(i) << ' ';
  std::cout << std::endl;

  // this implements a "sort" which is O(N * R) where (R) is the total number of
  // unique destination ranks. it performs better than other algorithms in the
  // case when (R) is small, but results may vary
  using DeviceType = typename InputView::traits::device_type;
  using ExecutionSpace = typename InputView::traits::execution_space;

  Kokkos::View<int *, DeviceType> device_ranks_duplicate(
      Kokkos::ViewAllocateWithoutInitializing(ranks.label()), ranks.size());
  Kokkos::deep_copy(device_ranks_duplicate, ranks);
  Kokkos::View<int *, DeviceType> device_permutation_indices(
      Kokkos::ViewAllocateWithoutInitializing(permutation_indices.label()),
      permutation_indices.size());
  Kokkos::View<int, DeviceType> device_offset("offset");
  while (true)
  {
    int const largest_rank = ArborX::max(device_ranks_duplicate);
    if (largest_rank == -1)
      break;
    unique_ranks.push_back(largest_rank);
    auto device_largest_rank = Kokkos::View<int, DeviceType>("largest_rank");
    Kokkos::deep_copy(device_largest_rank, largest_rank);
    Kokkos::parallel_scan(
        "process_biggest_rank_items", Kokkos::RangePolicy<ExecutionSpace>(0, n),
        KOKKOS_LAMBDA(int i, int &update, bool last_pass) {
          bool const is_largest_rank =
              (device_ranks_duplicate(i) == device_largest_rank());
          if (is_largest_rank)
          {
            if (last_pass)
            {
              device_permutation_indices(i) = update + device_offset();
              device_ranks_duplicate(i) = -1;
              //sleep(1);
            }
	              ExecutionSpace().fence();
	    //sleep(1);
            ++update;
	              ExecutionSpace().fence();
            sleep(1);
          }
	  ExecutionSpace().fence();
//	  sleep(1);
          if (last_pass && i + 1 == device_ranks_duplicate.extent(0))
            device_offset() += update;
        });
    ExecutionSpace().fence();
    //sleep(10);
    auto offset =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_offset);
    offsets.push_back(offset());
  }
  ExecutionSpace().fence();
  //sleep(10);
  counts.reserve(offsets.size() - 1);
  for (unsigned int i = 1; i < offsets.size(); ++i)
    counts.push_back(offsets[i] - offsets[i - 1]);
  //sleep(10);
  Kokkos::deep_copy(permutation_indices, device_permutation_indices);
  assert(offsets.back() == ranks.size());

  std::cout << "unique_ranks" << std::endl;
  for (const auto el: unique_ranks)
    std::cout << el << ' ';
  std::cout << std::endl;

  std::cout << "counts" << std::endl;  
  for (const auto el: counts)  
    std::cout << el << ' ';  
  std::cout << std::endl;  

std::cout << "offstes" << std::endl;  
  for (const auto el: offsets)  
    std::cout << el << ' ';  
  std::cout << std::endl; 

 std::cout << "permutation" << std::endl;
  for (unsigned int i=0; i<permutation_indices.size(); ++i)
    std::cout << permutation_indices(i) << ' ';
  std::cout << std::endl;

  std::vector <int> permutation_copy(permutation_indices.size());
	  for (unsigned int i=0; i<permutation_indices.size(); ++i)
		  permutation_copy[i] = permutation_indices(i);

  std::sort(permutation_copy.begin(), permutation_copy.end());
  for (unsigned int i=0; i<permutation_indices.size(); ++i)
  {
    if (!(permutation_copy[i]==i))
    {
      std::cout << permutation_copy[i] << " should be " << i << std::endl;
      assert(permutation_copy[i]==i);
    }
  }
}

class Distributor
{
public:
  Distributor(MPI_Comm comm)
      : _comm(comm)
  {
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

    _permute = Kokkos::View<int *, Kokkos::HostSpace>(
        Kokkos::ViewAllocateWithoutInitializing("permute"),
        destination_ranks.size());
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

#ifndef ARBORX_USE_CUDA_AWARE_MPI
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "");
#endif

    Kokkos::View<ValueType *, typename View::traits::device_type> dest_buffer(
        Kokkos::ViewAllocateWithoutInitializing("destination_buffer"),
        exports.size());

    Kokkos::View<int *, typename View::traits::device_type> permute_mirror(
        Kokkos::ViewAllocateWithoutInitializing("permute_device_mirror"),
        _permute.size());
    Kokkos::deep_copy(permute_mirror, _permute);

    Kokkos::parallel_for("copy_destinations_permuted",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             0, _dest_offsets.back() * num_packets),
                         KOKKOS_LAMBDA(int const k) {
                           int const i = k / num_packets;
                           int const j = k % num_packets;
                           dest_buffer(num_packets * permute_mirror[i] + j) =
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
  Kokkos::View<int *, Kokkos::HostSpace> _permute;
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
