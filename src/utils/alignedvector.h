#pragma once

#include <vector>
#include <boost/align/aligned_allocator.hpp>

/*
 * Vector with customizable allocator
 * https://stackoverflow.com/questions/55209596/is-it-possible-to-have-a-stdvectorchar-allocate-memory-with-a-chosen-memory
 */
template <typename T, size_t align>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, align>>;
