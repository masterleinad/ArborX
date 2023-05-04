/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_BOOST_TEST_COMPRESSED_STORAGE_COMPARISON_HPP
#define ARBORX_BOOST_TEST_COMPRESSED_STORAGE_COMPARISON_HPP

#include <boost/test/tools/detail/print_helper.hpp>
#include <boost/test/utils/is_forward_iterable.hpp>

#include <Kokkos_Core.hpp>

#include <iosfwd>
#include <set>

template <typename Values>
struct ValueComparator;// : std::less<Values>{};

template<>
struct ValueComparator<int> : std::less<int> {};

template<>
struct ValueComparator<Kokkos::pair<int, float>>
{
  bool operator()(const Kokkos::pair<int, float> left, const Kokkos::pair<int, float> right) const
  {
    if (left.first >= right.first)
    {
	                  std::cout << "False " << left.first << ' ' << left.second << " and " << right.first << ' ' << right.second << std::endl;
	    return false;
    }
    if (left.second >= right.second * (1. - 1.e-3))
    {
	                              std::cout << "False " << left.first << ' ' << left.second << " and " << right.first << ' ' << right.second << std::endl;
	    return false;
    }
    std::cout << "True " << left.first << ' ' << left.second << " and " << right.first << ' ' << right.second << std::endl;
    return true;
  }
};

template <typename Offsets, typename Values>
struct CompressedStorage
{
  Offsets offsets;
  Values values;
  using value_type = typename Values::value_type;
  using index_type = typename Offsets::value_type;
  using comparator_type = ValueComparator<value_type>;
  struct ConstForwardIterator
  {
    index_type i;
    CompressedStorage const *p;
    using value_type = std::multiset<CompressedStorage::value_type, comparator_type>;
    ConstForwardIterator &operator++()
    {
      ++i;
      return *this;
    }
    ConstForwardIterator operator++(int)
    {
      ConstForwardIterator old{*this};
      ++*this;
      return old;
    };
    bool operator==(ConstForwardIterator const &o)
    {
      return i == o.i;
    }
    bool operator!=(ConstForwardIterator const &o)
    {
      return !(*this == o);
    }
    value_type operator*()
    {
      return {p->values.data() + p->offsets[i],
              p->values.data() + p->offsets[i + 1], comparator_type{}};
    }
  };
  ConstForwardIterator cbegin() const { return {0, this}; }
  ConstForwardIterator cend() const { return {static_cast<index_type>(offsets.size()), this}; }
  std::size_t size() const { return offsets.size() - 1; }
};

template <typename Offsets, typename Values>
CompressedStorage<std::decay_t<Offsets>, std::decay_t<Values>>
make_compressed_storage(Offsets &&offsets, Values &&values)
{
  return {std::forward<Offsets>(offsets), std::forward<Values>(values)};
}

namespace boost
{
namespace unit_test
{
template <typename Offsets, typename Values>
struct is_forward_iterable<CompressedStorage<Offsets, Values>>
    : public boost::mpl::true_
{};
template <typename Offsets, typename Values>
struct bt_iterator_traits<CompressedStorage<Offsets, Values>, true>
{
  using this_type = CompressedStorage<Offsets, Values>;
  using const_iterator = typename this_type::ConstForwardIterator;
  using value_type = typename const_iterator::value_type;
  static const_iterator begin(this_type const &x) { return x.cbegin(); }
  static const_iterator end(this_type const &x) { return x.cend(); }
  static std::size_t size(this_type const &x) { return x.size(); }
};
} // namespace unit_test
} // namespace boost

// Customization for logging with Boost.Test
namespace std
{
template <typename Key, typename Compare, typename Allocator>
std::ostream &
boost_test_print_type(std::ostream &os,
                      std::multiset<Key, Compare, Allocator> const &s)
{
  os << '(';
  for (auto const &x : s)
  {
    os << ' ';
    boost::test_tools::tt_detail::print_log_value<Key>()(os, x);
  }
  os << " )";
  return os;
}
} // namespace std

#endif
