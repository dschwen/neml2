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

#include "DanielFlowRate.h"

namespace neml2
{
register_NEML2_object(DanielFlowRate);

ParameterSet
DanielFlowRate::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<CrossRef<Scalar>>("parameter_1");
  params.set<CrossRef<Scalar>>("parameter_2");
  params.set<CrossRef<Scalar>>("parameter_3");
  params.set<CrossRef<Scalar>>("parameter_4");
  params.set<LabeledAxisAccessor>("mandel_stress") = {{"state", "internal", "M"}};
  params.set<LabeledAxisAccessor>("flow_rate") = {{"state", "internal", "gamma_rate"}};
  params.set<LabeledAxisAccessor>("temperature") = {{"forces", "T"}};
  return params;
}

DanielFlowRate::DanielFlowRate(const ParameterSet & params)
  : Model(params),
    mandel_stress(declare_input_variable<SymR2>(params.get<LabeledAxisAccessor>("mandel_stress"))),
    flow_rate(declare_output_variable<Scalar>(params.get<LabeledAxisAccessor>("flow_rate"))),
    temperature(declare_input_variable<Scalar>(params.get<LabeledAxisAccessor>("temperature"))),
    _p1(register_crossref_model_parameter<Scalar>("p1", "parameter_1")),
    _p2(register_crossref_model_parameter<Scalar>("p2", "parameter_2")),
    _p3(register_crossref_model_parameter<Scalar>("p3", "parameter_3"))
{
  setup();
}

void
DanielFlowRate::set_value(const LabeledVector & in,
                          LabeledVector * out,
                          LabeledMatrix * dout_din,
                          LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "I am too lazy to implement second derivatives");
  const auto options = in.options();

  // Grab the mandel stress and temperature
  auto M = in.get<SymR2>(mandel_stress);
  auto T = in.get<Scalar>(temperature);

  // Let's say the flow resistance increases as temperature increases
  auto eta = _p2 * T;

  // Compute Daniel's flow rate
  auto S = M.dev();
  Scalar vm = std::sqrt(3.0 / 2.0) * S.norm(EPS);
  auto f = vm - _p1;
  Scalar Hf = math::heaviside(f);
  Scalar f_abs = torch::abs(f);
  Scalar gamma_dot_m = torch::pow(f_abs / eta, _p3);
  Scalar gamma_dot = gamma_dot_m * Hf;

  if (out)
    out->set(gamma_dot, flow_rate);

  if (dout_din)
  {
    // Derivative of Daniel's flow rate
    auto dgamma_dot_df = _p3 / f_abs * gamma_dot;
    auto df_dvm = Scalar::identity_map(options);
    auto dvm_dM = 3.0 / 2.0 * S / vm;
    auto dgamma_dot_dM = dgamma_dot_df * df_dvm * dvm_dM;
    dout_din->set(dgamma_dot_dM, flow_rate, mandel_stress);
    dout_din->set(-gamma_dot * _p3 / eta * _p2, flow_rate, temperature);
  }
}
} // namespace neml2
