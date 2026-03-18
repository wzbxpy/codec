#pragma once

#include <type_traits>

template <typename F>
__host__ __device__ constexpr void
bool_switch(bool expr, F&& f) {
    if (expr) {
        std::forward<F>(f)(std::true_type{});
    } else {
        std::forward<F>(f)(std::false_type{});
    }
}

template <typename T, T... Vs, class Case, class Default>
__host__ __device__ constexpr void
value_switch(T value, Case&& case_fn, Default&& default_fn) {
    bool match = ((value == Vs
                       ? (static_cast<void>(std::forward<Case>(case_fn)(std::integral_constant<T, Vs>{})), true)
                       : false) ||
                  ...);
    if (!match) {
        std::forward<Default>(default_fn)();
    }
}

namespace dbg {
template <class>
struct type_printer;
}
