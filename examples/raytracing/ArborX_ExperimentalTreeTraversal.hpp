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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_Ray.hpp>

namespace ArborX
{
namespace Experimental
{

template <class BVH, class Predicates, class Callback>
struct PriorityBasedTreeTraversal
{
  BVH _bvh;
  Predicates _predicates;
  Callback _callback;

  using Access = AccessTraits<Predicates, PredicatesTag>;

  template <class ExecutionSpace>
  PriorityBasedTreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                             Predicates const &predicates,
                             Callback const &callback)
      : _bvh{bvh}
      , _predicates{predicates}
      , _callback{callback}
  {
    if (_bvh.empty())
    {
      // do nothing
    }
    else if (_bvh.size() == 1)
    {
      Kokkos::parallel_for("ArborX::Experimental::PriorityBasedTreeTraversal::"
                           "degenerated_one_leaf_tree",
                           Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
    else
    {
      Kokkos::parallel_for("ArborX::Experimental::PriorityBasedTreeTraversal",
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION void operator()(OneLeafTree, int predicate_index) const
  {
    auto const &predicate = Access::get(_predicates, predicate_index);
    using ArborX::Details::HappyTreeFriends;
    auto const root = HappyTreeFriends::getRoot(_bvh);
    auto const &root_bounding_volume =
        HappyTreeFriends::getBoundingVolume(_bvh, root);
    if (predicate(root_bounding_volume))
    {
      _callback(predicate, 0);
    }
  }

  KOKKOS_FUNCTION void operator()(int predicate_index) const
  {
    auto const &predicate = Access::get(_predicates, predicate_index);
    using ArborX::Details::HappyTreeFriends;
    using ArborX::Details::invoke_callback_and_check_early_exit;

    auto const distance = [ray = getGeometry(predicate), bvh = _bvh](int node)
    {
      float tmin;
      float tmax;
      auto const &box = HappyTreeFriends::getBoundingVolume(bvh, node);
      bool const ray_intersects_box = intersection(ray, box, tmin, tmax);
      assert(ray_intersects_box);
      (void)ray_intersects_box;
      return tmin;
    };

    using PairIndexDistance = Kokkos::pair<int, float>;
    struct CompareDistance
    {
      KOKKOS_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                      PairIndexDistance const &rhs) const
      {
        return lhs.second > rhs.second;
      }
    };

    PairIndexDistance heap[64];
    PairIndexDistance *heap_last = heap;
    CompareDistance const compare;

    using ArborX::Details::popHeap;
    using ArborX::Details::pushHeap;
    int node = HappyTreeFriends::getRoot(_bvh);
    int left_child;
    int right_child;
    float distance_left;
    float distance_right;

    while (true)
    {
      bool traverse_left = false;
      bool traverse_right = false;

      if (HappyTreeFriends::isLeaf(_bvh, node))
      {
        if (invoke_callback_and_check_early_exit(
                _callback, predicate,
                HappyTreeFriends::getLeafPermutationIndex(_bvh, node)))
          return;
      }
      else
      {
        left_child = HappyTreeFriends::getLeftChild(_bvh, node);
        right_child = HappyTreeFriends::getRightChild(_bvh, node);

        if (predicate(HappyTreeFriends::getBoundingVolume(_bvh, left_child)))
        {
          distance_left = distance(left_child);
          traverse_left = true;
        }

        if (predicate(HappyTreeFriends::getBoundingVolume(_bvh, right_child)))
        {
          distance_right = distance(right_child);
          traverse_right = true;
        }
      }

      if (!traverse_left && !traverse_right)
      {
        if (heap != heap_last)
        {
          node = heap->first;
          popHeap(heap, heap_last--, compare);
        }
        else
        {
          break; // heap is empty
        }
      }
      else
      {
        node = (traverse_left &&
                (distance_left <= distance_right || !traverse_right))
                   ? left_child
                   : right_child;
        if (traverse_left && traverse_right)
        {
          *heap_last++ = node == left_child
                             ? Kokkos::make_pair(right_child, distance_right)
                             : Kokkos::make_pair(left_child, distance_left);
          pushHeap(heap, heap_last, compare);
        }
      }
    }
  }
};

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
void traverse(ExecutionSpace const &space, BVH const &bvh,
              Predicates const &predicates, Callback const &callback)
{
  PriorityBasedTreeTraversal<BVH, Predicates, Callback>(space, bvh, predicates,
                                                        callback);
}
} // namespace Experimental
} // namespace ArborX
