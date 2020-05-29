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
#ifndef ARBORX_DETAILS_BUFFER_OPTIMIZATON_HPP
#define ARBORX_DETAILS_BUFFER_OPTIMIZATON_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

namespace ArborX
{
namespace Details
{

enum BufferStatus
{
  PreallocationNone = 0,
  PreallocationHard = -1,
  PreallocationSoft = 1
};

BufferStatus toBufferStatus(int buffer_size)
{
  if (buffer_size == 0)
    return BufferStatus::PreallocationNone;
  if (buffer_size > 0)
    return BufferStatus::PreallocationSoft;
  return BufferStatus::PreallocationHard;
}

struct FirstPassTag
{
};
struct FirstPassNoBufferOptimizationTag
{
};
struct SecondPassTag
{
};

template <typename PassTag, typename Predicates, typename Callback,
          typename OutputView, typename CountView, typename OffsetView,
          typename PermuteType>
struct InsertGenerator
{
  Predicates _permuted_predicates;
  Callback _callback;
  OutputView _out;
  CountView _counts;
  OffsetView _offset;
  PermuteType _permute;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using Tag = typename Traits::Helper<Access>::tag;

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, SpatialPredicateTag>{}>
  operator()(int predicate_index, int primitive_index) const
  {
    auto const permuted_predicate_index = _permute(predicate_index);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size. For this reason, also take a reference for offset.
    auto const &offset = _offset(permuted_predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                if (count_old < buffer_size)
                  _out(offset + count_old) = value;
              });
  }
  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(int predicate_index, int primitive_index, float distance) const
  {
    auto const permuted_predicate_index = _permute(predicate_index);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size. For this reason, also take a reference for offset.
    auto const &offset = _offset(permuted_predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                if (count_old < buffer_size)
                  _out(offset + count_old) = value;
              });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, SpatialPredicateTag>{}>
      operator()(int predicate_index, int primitive_index) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index,
              [&](ValueType const &) { Kokkos::atomic_fetch_add(&count, 1); });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, NearestPredicateTag>{}>
      operator()(int predicate_index, int primitive_index, float distance) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance,
              [&](ValueType const &) { Kokkos::atomic_fetch_add(&count, 1); });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                                   std::is_same<V, SpatialPredicateTag>{}>
  operator()(int predicate_index, int primitive_index) const
  {
    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    // TODO: there is a tradeoff here between skipping computation offset +
    // count, and atomic increment of count. I think atomically incrementing
    // offset is problematic for OpenMP as you potentially constantly steal
    // cache lines.
    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, [&](ValueType const &value) {
                //_out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
                 _out.insert(
                    Kokkos::pair<int, int>{predicate_index, Kokkos::atomic_fetch_add(&offset, 1)},
                    value);
              });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(int predicate_index, int primitive_index, float distance) const
  {
    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    // TODO: there is a tradeoff here between skipping computation offset +
    // count, and atomic increment of count. I think atomically incrementing
    // offset is problematic for OpenMP as you potentially constantly steal
    // cache lines.
    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance, [&](ValueType const &value) {
                //_out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
                 _out.insert(
                    Kokkos::pair<int, int>{predicate_index, Kokkos::atomic_fetch_add(&offset, 1)},
                    value);
              });
  }
};

template <typename Predicates, typename Permute>
struct PermutedPredicates
{
  Predicates _predicates;
  Permute _permute;
  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return _predicates(_permute(i));
  }
};

} // namespace Details

namespace Traits
{
template <typename Predicates, typename Permute>
struct Access<Details::PermutedPredicates<Predicates, Permute>, PredicatesTag>
{
  using PermutedPredicates = Details::PermutedPredicates<Predicates, Permute>;
  using NativeAccess = Access<Predicates, PredicatesTag>;

  inline static std::size_t size(PermutedPredicates const &permuted_predicates)
  {
    return NativeAccess::size(permuted_predicates._predicates);
  }

  KOKKOS_INLINE_FUNCTION static auto
  get(PermutedPredicates const &permuted_predicates, std::size_t i)
  {
    return NativeAccess::get(permuted_predicates._predicates,
                             permuted_predicates._permute(i));
  }
  using memory_space = typename NativeAccess::memory_space;
};
} // namespace Traits

namespace Details
{

template <typename ExecutionSpace, typename TreeTraversal, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView,
          typename PermuteType>
void queryImpl(ExecutionSpace const &space, TreeTraversal const &tree_traversal,
               Predicates const &predicates, Callback const &callback,
               OutputView &out, OffsetView &offset, PermuteType permute,
               BufferStatus buffer_status)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  using MapType = /*OutputView;*/
      Kokkos::UnorderedMap<Kokkos::pair<int, int>,
                           typename OutputView::value_type, ExecutionSpace>;
  MapType unordered_map(1000000);

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass");

  using CountView = Kokkos::View<int *, ExecutionSpace>;
  CountView counts(Kokkos::view_alloc("counts", space), n_queries);

  using PermutedPredicates = PermutedPredicates<Predicates, PermuteType>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:first_pass");
  bool underflow = false;
  bool overflow = false;
  {
    InsertGenerator<FirstPassNoBufferOptimizationTag, PermutedPredicates,
                    Callback, MapType, CountView, OffsetView, PermuteType>
        insert_generator{permuted_predicates,
                         callback,
                         unordered_map,
                         counts,
                         offset,
                         permute};
    tree_traversal.launch(space, permuted_predicates, insert_generator);
    // This may not be true, but it does not matter. As long as we have
    // (n_results == 0) check before second pass, this value is not used.
    // Otherwise, we know it's overflowed as there is no allocation.
    overflow = true;
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:first_pass_postprocess");

  OffsetView preallocated_offset("offset_copy", 0);
  if (underflow)
  {
    // Store a copy of the original offset. We'll need it for compression.
    preallocated_offset = clone(space, offset);
  }

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("copy_counts_to_offsets"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int const i) { offset(permute(i)) = counts(i); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();

  if (n_results == 0)
  {
    // Exit early if either no results were found for any of the queries, or
    // nothing was inserted inside a callback for found results. This check
    // guarantees that the second pass will not be executed.
    Kokkos::resize(out, 0);
    // FIXME: do we need to reset offset if it was preallocated here?
    Kokkos::Profiling::popRegion();
    return;
  }

  {
    // Not enough (individual) storage for results

    // If it was hard preallocation, we simply throw
    ARBORX_ASSERT(buffer_status != BufferStatus::PreallocationHard);

    // Otherwise, do the second pass
    Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:second_pass");

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("copy_offsets_to_counts"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int const i) { counts(i) = offset(permute(i)); });

    // reallocWithoutInitializing(out, n_results);

    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<SecondPassTag, PermutedPredicates, Callback, MapType,
                        CountView, OffsetView, PermuteType>{
            permuted_predicates, callback, unordered_map, counts, offset,
            permute});

    // fill the output view from the unordered_map
    Kokkos::parallel_for(unordered_map.capacity(), KOKKOS_LAMBDA (uint32_t i) {
    if( unordered_map.valid_at(i) ) {
    auto key   = unordered_map.key_at(i);
    auto value = unordered_map.value_at(i);
    out(/*offset(key.first)+*/key.second) = value;
     }
});


    Kokkos::Profiling::popRegion();
  }
  Kokkos::Profiling::popRegion();
} // namespace Details

} // namespace Details
} // namespace ArborX

#endif
