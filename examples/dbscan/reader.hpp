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

#ifndef READER_HPP
#define READER_HPP

#include <ArborX_Point.hpp>

#include <string>
#include <vector>

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    std::string const &reader_type = "arborx",
                                    bool binary = false);

#endif
