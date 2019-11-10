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

#include <ArborX_DetailsDistributedSearchTreeImpl.hpp>
#include <ArborX_DetailsUtils.hpp> // max
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp> // FIXME

#include <algorithm> // max_element
#include <cassert>
#include <memory>
#include <numeric> // iota
#include <sstream>
#include <vector>

#include <mpi.h>

namespace ArborX
{

namespace internal
{
template <typename PointerType>
struct PointerDepth
{
  static int constexpr value = 0;
};

template <typename PointerType>
struct PointerDepth<PointerType *>
{
  static int constexpr value = PointerDepth<PointerType>::value + 1;
};

template <typename PointerType, std::size_t N>
struct PointerDepth<PointerType[N]>
{
  static int constexpr value = PointerDepth<PointerType>::value;
};
} // namespace internal

template <typename View>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename View::traits::host_mirror_space>
create_layout_right_mirror_view(
    View const &src,
    typename std::enable_if<!(
        (std::is_same<typename View::traits::array_layout,
                      Kokkos::LayoutRight>::value ||
         (View::rank == 1 && !std::is_same<typename View::traits::array_layout,
                                           Kokkos::LayoutStride>::value)) &&
        std::is_same<typename View::traits::memory_space,
                     typename View::traits::host_mirror_space::memory_space>::
            value)>::type * = 0)
{
  constexpr int pointer_depth =
      internal::PointerDepth<typename View::traits::data_type>::value;
  return Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                      typename View::traits::host_mirror_space>(
      std::string(src.label()).append("_layout_right_mirror"), src.extent(0),
      pointer_depth > 1 ? src.extent(1) : KOKKOS_INVALID_INDEX,
      pointer_depth > 2 ? src.extent(2) : KOKKOS_INVALID_INDEX,
      pointer_depth > 3 ? src.extent(3) : KOKKOS_INVALID_INDEX,
      pointer_depth > 4 ? src.extent(4) : KOKKOS_INVALID_INDEX,
      pointer_depth > 5 ? src.extent(5) : KOKKOS_INVALID_INDEX,
      pointer_depth > 6 ? src.extent(6) : KOKKOS_INVALID_INDEX,
      pointer_depth > 7 ? src.extent(7) : KOKKOS_INVALID_INDEX);
}

template <typename View>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename View::traits::host_mirror_space>
create_layout_right_mirror_view(
    View const &src,
    typename std::enable_if<
        ((std::is_same<typename View::traits::array_layout,
                       Kokkos::LayoutRight>::value ||
          (View::rank == 1 && !std::is_same<typename View::traits::array_layout,
                                            Kokkos::LayoutStride>::value)) &&
         std::is_same<typename View::traits::memory_space,
                      typename View::traits::host_mirror_space::memory_space>::
             value)>::type * = 0)
{
  return src;
}

namespace Details
{

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

  template <typename View, typename... OtherViews>
  typename std::enable_if<Kokkos::is_view<View>::value>::type
  sendAcrossNetwork(View exports, typename View::non_const_type imports,
                    OtherViews... other_views) const
  {
    ARBORX_ASSERT((exports.extent(0) == this->getTotalSendLength()) &&
                  (imports.extent(0) == this->getTotalReceiveLength()) &&
                  (exports.extent(1) == imports.extent(1)) &&
                  (exports.extent(2) == imports.extent(2)) &&
                  (exports.extent(3) == imports.extent(3)) &&
                  (exports.extent(4) == imports.extent(4)) &&
                  (exports.extent(5) == imports.extent(5)) &&
                  (exports.extent(6) == imports.extent(6)) &&
                  (exports.extent(7) == imports.extent(7)));

    auto const num_packets = exports.extent(1) * exports.extent(2) *
                             exports.extent(3) * exports.extent(4) *
                             exports.extent(5) * exports.extent(6) *
                             exports.extent(7);

#ifndef ARBORX_USE_CUDA_AWARE_MPI
    auto exports_host = create_layout_right_mirror_view(exports);
    Kokkos::deep_copy(exports_host, exports);

    auto imports_host = create_layout_right_mirror_view(imports);

    using NonConstValueType = typename View::non_const_value_type;
    using ConstValueType = typename View::const_value_type;

    Kokkos::View<ConstValueType *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        export_buffer(exports_host.data(), exports_host.size());

    Kokkos::View<NonConstValueType *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        import_buffer(imports_host.data(), imports_host.size());

    auto permuted_exports = permute_source(export_buffer);
    auto mpi_requests =
        this->doPostsAndWaits(permuted_exports, num_packets, import_buffer);
    this->sendAcrossNetwork(other_views...);
    for (auto &request : mpi_requests)
      MPI_Wait(request.get(), MPI_STATUS_IGNORE);

    Kokkos::deep_copy(imports, imports_host);
#else
    auto permuted_exports = permute_source(exports);
    auto mpi_requests =
        this->doPostsAndWaits(permuted_exports, num_packets, imports);
    this->sendAcrossNetwork(other_views...);
    for (auto &request : mpi_requests)
      MPI_Wait(request.get(), MPI_STATUS_IGNORE);
#endif
  }

  void sendAcrossNetwork() const {}

  template <typename View>
  Kokkos::View<typename View::value_type *, typename View::traits::device_type>
  permute_source(View const &source) const
  {
    using ValueType = typename View::non_const_value_type;
    using ExecutionSpace = typename View::execution_space;
    static_assert(View::rank == 1, "");

#ifndef ARBORX_USE_CUDA_AWARE_MPI
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "");
#endif

    Kokkos::View<ValueType *, typename View::traits::device_type>
        permuted_source(
            Kokkos::ViewAllocateWithoutInitializing("destination_buffer"),
            source.size());

    if (_dest_offsets.back() == 0)
    {
      ARBORX_ASSERT(source.size() == 0);
      return permuted_source;
    }

    ARBORX_ASSERT(source.size() % _dest_offsets.back() == 0);
    auto const num_packets = source.size() / _dest_offsets.back();

    auto permute_mirror = Kokkos::create_mirror_view_and_copy(
        typename View::traits::device_type(), _permute);

    Kokkos::parallel_for("copy_destinations_permuted",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             0, _dest_offsets.back() * num_packets),
                         KOKKOS_LAMBDA(int const k) {
                           int const i = k / num_packets;
                           int const j = k % num_packets;
                           permuted_source(num_packets * permute_mirror[i] +
                                           j) = source[num_packets * i + j];
                         });

    return permuted_source;
  }

  template <typename View>
  std::vector<std::unique_ptr<MPI_Request>>
  doPostsAndWaits(typename View::const_type const &exports, size_t num_packets,
                  View const &imports) const
  {
    ARBORX_ASSERT(num_packets * _dest_offsets.back() == exports.size());
    ARBORX_ASSERT(num_packets * _src_offsets.back() == imports.size());

    using ValueType = typename View::value_type;
    using ExecutionSpace = typename View::execution_space;
    static_assert(View::rank == 1, "");

#ifndef ARBORX_USE_CUDA_AWARE_MPI
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "");
#endif

    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);
    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    std::vector<std::unique_ptr<MPI_Request>> requests;
    requests.reserve(outdegrees + indegrees);
    for (int i = 0; i < indegrees; ++i)
    {
      if (_sources[i] != comm_rank)
      {
        auto const message_size =
            _src_counts[i] * num_packets * sizeof(ValueType);
        auto const receive_buffer_ptr =
            imports.data() + _src_offsets[i] * num_packets;
        requests.emplace_back(new MPI_Request);
        int const ierr =
            MPI_Irecv(receive_buffer_ptr, message_size, MPI_BYTE, _sources[i],
                      123, _comm, requests.back().get());
        ARBORX_ASSERT(ierr == MPI_SUCCESS);
      }
    }

    // make sure the data in dest_buffer has been copied before sending it.
    ExecutionSpace().fence();

    for (int i = 0; i < outdegrees; ++i)
    {
      auto const message_size =
          _dest_counts[i] * num_packets * sizeof(ValueType);
      auto const send_buffer_ptr =
          exports.data() + _dest_offsets[i] * num_packets;
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
        requests.emplace_back(new MPI_Request);
        int const ierr =
            MPI_Isend(send_buffer_ptr, message_size, MPI_BYTE, _destinations[i],
                      123, _comm, requests.back().get());
        ARBORX_ASSERT(ierr == MPI_SUCCESS);
      }
    }
    return requests;
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
