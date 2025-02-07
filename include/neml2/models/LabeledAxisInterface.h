// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/tensors/LabeledAxis.h"

namespace neml2
{
class LabeledAxisInterface
{
public:
  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

protected:
  /// Declare an axis
  [[nodiscard]] LabeledAxis & declareAxis();

  /// Declare an item recursively on an axis
  template <typename T>
  LabeledAxisAccessor declare_variable(LabeledAxis & axis,
                                       const std::vector<std::string> & names) const
  {
    return declare_variable(axis, utils::storage_size(T::_base_sizes), names);
  }

  /// Declare an item (with known storage size) recursively on an axis
  LabeledAxisAccessor
  declare_variable(LabeledAxis & axis, TorchSize sz, const std::vector<std::string> & names) const
  {
    LabeledAxisAccessor accessor{names};
    axis.add(accessor, sz);
    return accessor;
  }

private:
  /// All the declared axes
  std::vector<std::shared_ptr<LabeledAxis>> _axes;
};
} // namespace neml2
