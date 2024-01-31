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

OptionSet
DanielFlowRate::expected_options()
{
  using vecstr = std::vector<std::string>;
  auto options = Model::expected_options();
  options.set<LabeledAxisAccessor>("trial_effective_stress") = vecstr{"state", "internal", "s"};
  // options.set<LabeledAxisAccessor>("dt") = vecstr{"forces", "dt"};
  options.set<LabeledAxisAccessor>("flow_rate") = vecstr{"state", "internal", "gamma_rate"};
  options.set<LabeledAxisAccessor>("temperature") = vecstr{"forces", std::string("T")};
  options.set<LabeledAxisAccessor>("grain_size") = vecstr{"forces", "grain_size"};
  options.set<LabeledAxisAccessor>("stoichiometry") = vecstr{"forces", "stoichiometry"};
  // options.set<LabeledAxisAccessor>("fission_rate") = vecstr{"forces", "fission_rate"};
  options.set<bool>("use_AD_first_derivative") = true;
  options.set<bool>("use_AD_second_derivative") = true;
  options.set<std::string>("model_file_name") = "model.pt";
  return options;
}

DanielFlowRate::DanielFlowRate(const OptionSet & options)
  : Model(options),
    trial_effective_stress(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("trial_effective_stress"))),
    // dt(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("dt"))),
    flow_rate(declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("flow_rate"))),
    temperature(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("temperature"))),
    grain_size(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("grain_size"))),
    stoichiometry(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("stoichiometry"))),
    // fission_rate(declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("fission_rate"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(
        torch::jit::load(options.get<std::string>("model_file_name")))),
    _x_mean(declare_buffer<BatchTensor>(
        "x_mean",
        BatchTensor(torch::tensor({1.850000e+03, 5.000000e-05, 5.000000e+07, 2.034279e-03}), 0))),
    _x_std(declare_buffer<BatchTensor>(
        "x_std",
        BatchTensor(torch::tensor({2.049391e+02, 2.886175e-05, 2.886175e+07, 2.750995e-03}), 0))),
    _y_mean(declare_buffer<Scalar>("y_mean", Scalar(-1.448253e+01, default_tensor_options()))),
    _y_std(declare_buffer<Scalar>("y_std", Scalar(5.293281e+00, default_tensor_options())))
{
  setup();

  // 1. We don't need the parameter gradients
  // 2. We need the parameters to have the same options as ours
  _surrogate->to(torch::kFloat64);
  for (auto param : _surrogate->parameters(/*recursive=*/true))
    param.requires_grad_(false);
}

void
DanielFlowRate::set_value(const LabeledVector & in,
                          LabeledVector * out,
                          LabeledMatrix * dout_din,
                          LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "I am too lazy to implement second derivatives");
  neml_assert_dbg(!dout_din, "Try AD");

  // Grab the trial stress, temperature, grain size, and stoichiometry
  // These are the inputs to the neural network
  auto s = in.get<Scalar>(trial_effective_stress);
  auto T = in.get<Scalar>(temperature);
  auto G = in.get<Scalar>(grain_size);
  auto ST = in.get<Scalar>(stoichiometry);

  // Compute Daniel's flow rate
  auto x = BatchTensor(torch::transpose(torch::vstack({T, G, s, ST}), 0, 1), in.batch_dim());
  auto x0 = (x - _x_mean) / _x_std;
  auto gamma0_dot = Scalar(_surrogate->forward({x0}).toTensor().squeeze(), in.batch_dim());
  auto gamma_dot = math::exp(gamma0_dot * _y_std + _y_mean);

  // // irradiation creep
  // auto c3 = _a7 * fission_rate;
  // auto activation_term3 = std::exp(-_q3 / _temperature[_qp]);
  // auto term_3 = c3 * activation_term3;

  if (out)
    out->set(gamma_dot, flow_rate);
}

void
DanielFlowRate::to(const torch::TensorOptions & options)
{
  Model::to(options);
  _surrogate->to(options.device());
}

} // namespace neml2
