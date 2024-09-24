// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/ScalarTrialStressUpdate.h"

namespace neml2
{
register_NEML2_object(ScalarTrialStressUpdate);

OptionSet
ScalarTrialStressUpdate::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set_input("elastic_trial_stress") = VariableName("forces", "s0");
  options.set("elastic_trial_stress").doc() = "Initial elastic trial stress";

  options.set_input("inelastic_strain") = VariableName("state", "ep");
  options.set("inelastic_strain").doc() = "Current guess for the inelastic strain rate";

  options.set_output("updated_trial_stress") = VariableName("state", "s");
  options.set("updated_trial_stress").doc() = "Trial stress corrected for inelastic deformation";

  options.set_parameter<CrossRef<Scalar>>("youngs_modulus");
  options.set("youngs_modulus").doc() = "Young's modulus";

  options.set_parameter<CrossRef<Scalar>>("poisson_ratio");
  options.set("poisson_ratio").doc() = "Poisson's ratio";
  return options;
}

ScalarTrialStressUpdate::ScalarTrialStressUpdate(const OptionSet & options)
  : Model(options),
    _elastic_trial_stress(declare_input_variable<Scalar>("elastic_trial_stress")),
    _inelastic_strain(declare_input_variable<Scalar>("inelastic_strain")),
    _inelastic_strain_old(declare_input_variable<Scalar>(_inelastic_strain.name().old())),
    _updated_trial_stress(declare_output_variable<Scalar>("updated_trial_stress")),
    _E(declare_parameter<Scalar>("E", "youngs_modulus", /*allow_nonlinear=*/true)),
    _nu(declare_parameter<Scalar>("nu", "poisson_ratio", /*allow_nonlinear=*/true))
{
}

void
ScalarTrialStressUpdate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto three_shear = 3.0 * _E / (2.0 * (1.0 + _nu));

  if (out)
    _updated_trial_stress =
        _elastic_trial_stress - three_shear * (_inelastic_strain - _inelastic_strain_old);

  if (dout_din)
  {
    if (_elastic_trial_stress.is_dependent())
      _updated_trial_stress.d(_elastic_trial_stress) = Scalar::ones(options()).base_expand_as(
          _updated_trial_stress.d(_elastic_trial_stress).value());

    if (_inelastic_strain.is_dependent())
      _updated_trial_stress.d(_inelastic_strain) = -three_shear;
  }

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
