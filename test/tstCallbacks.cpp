#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_Predicates.hpp>

struct NearestPredicates
{
};

struct SpatialPredicates
{
};

namespace ArborX
{
template <>
struct AccessTraits<NearestPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(NearestPredicates const &) { return 1; }
  static auto get(NearestPredicates const &, int) { return nearest(Point{}); }
};
template <>
struct AccessTraits<SpatialPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(SpatialPredicates const &) { return 1; }
  static auto get(SpatialPredicates const &, int)
  {
    return intersects(Point{});
  }
};
} // namespace ArborX

// Custom callbacks
struct SpatialPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, OutputFunctor const &) const
  {
  }
};

struct NearestPredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, ArborX::Details::DistanceReturnType,
                  OutputFunctor const &) const
  {
  }
};

struct Wrong
{
};

struct SpatialPredicateCallbackDoesNotTakeCorrectArgument
{
  template <typename OutputFunctor>
  void operator()(Wrong, int, OutputFunctor const &) const
  {
  }
};

int main()
{
  using ArborX::Details::check_valid_callback;

  // view type does not matter as long as we do not call the output functor
  Kokkos::View<ArborX::Details::DistanceReturnType *> v;

  // check_valid_callback(ArborX::Details::CallbackDefaultSpatialPredicate{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(ArborX::Details::CallbackDefaultNearestPredicate{},
  //                     NearestPredicates{}, v);

  // check_valid_callback(
  //    ArborX::Details::CallbackDefaultNearestPredicateWithDistance{},
  //    NearestPredicates{}, v);

  // not required to tag inline callbacks any more
  // check_valid_callback(SpatialPredicateCallbackMissingTag{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(NearestPredicateCallbackMissingTag{},
  //                     NearestPredicates{}, v);

  // generic lambdas are supported if not using NVCC
#ifndef __NVCC__
  // check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
  //                        auto const & /*out*/) {},
  //                     SpatialPredicates{}, v);

  // check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
  //                        ArborX::Details::DistanceReturnType /*distance*/,
  //                        auto const & /*out*/) {},
  //                     NearestPredicates{}, v);
#endif

  // Uncomment to see error messages

  // check_valid_callback(SpatialPredicateCallbackDoesNotTakeCorrectArgument{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(ArborX::Details::CallbackDefaultSpatialPredicate{},
  //                     NearestPredicates{}, v);

  // check_valid_callback(ArborX::Details::CallbackDefaultNearestPredicate{},
  //                     SpatialPredicates{}, v);
}
