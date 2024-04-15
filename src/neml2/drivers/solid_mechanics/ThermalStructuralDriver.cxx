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

#include "neml2/drivers/solid_mechanics/ThermalStructuralDriver.h"

namespace neml2
{
register_NEML2_object(ThermalStructuralDriver);

OptionSet
ThermalStructuralDriver::expected_options()
{
  auto options = SolidMechanicsDriver::expected_options();
  options.set<LabeledAxisAccessor>("temperature") =
      LabeledAxisAccessor{{"forces", std::string("T")}};
  options.set<CrossRef<torch::Tensor>>("prescribed_temperatures");
  return options;
}

ThermalStructuralDriver::ThermalStructuralDriver(const OptionSet & options)
  : SolidMechanicsDriver(options),
    _temperature_name(options.get<LabeledAxisAccessor>("temperature")),
    _temperature(options.get<CrossRef<torch::Tensor>>("prescribed_temperatures"), 2)
{
  check_integrity();
}

void
ThermalStructuralDriver::update_forces()
{
  SolidMechanicsDriver::update_forces();
  auto current_temperature = _temperature.batch_index({_step_count});
  _in.set(current_temperature, _temperature_name);
}

void
ThermalStructuralDriver::check_integrity() const
{
  neml_assert(_temperature.dim() == 2,
              "Input temperature should have dimension 2 but instead has dimension ",
              _temperature.dim());
  neml_assert(_time.sizes()[0] == _temperature.sizes()[0],
              "Input temperature and time should have the same number of time steps. The input "
              "time has ",
              _time.sizes()[0],
              " time steps, while the input time has ",
              _temperature.sizes()[0],
              " time steps");
  neml_assert(_time.sizes()[1] == _temperature.sizes()[1],
              "Input temperature and time should have the same batch size. The input time has a "
              "batch size of ",
              _time.sizes()[1],
              " while the input temperatue has a batch size of ",
              _temperature.sizes()[1]);
}
}
