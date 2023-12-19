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

#include "neml2/models/Model.h"
#include <torch/script.h>
#include <memory>

namespace neml2
{

class DanielFlowRate : public Model
{
public:
  static OptionSet expected_options();

  DanielFlowRate(const OptionSet & options);

  const LabeledAxisAccessor mandel_stress;
  const LabeledAxisAccessor flow_rate;

  const LabeledAxisAccessor temperature;
  const LabeledAxisAccessor grain_size;
  const LabeledAxisAccessor stoichiometry;

  virtual void to(const torch::Device & device) override;

protected:
  /// The flow rate
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;

  /// we need to use a pointer here because forward is not const qualified, :eye_roll:
  std::unique_ptr<torch::jit::script::Module> _surrogate;

  const BatchTensor & _x_mean;
  const BatchTensor & _x_std;
  const Scalar & _y_mean;
  const Scalar & _y_std;
};

} // namespace neml2
