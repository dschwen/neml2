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
  options.set<VariableName>("trial_effective_stress") = vecstr{"state", "internal", "s"};
  // options.set<VariableName>("dt") = vecstr{"forces", "dt"};
  options.set<VariableName>("flow_rate") = vecstr{"state", "internal", "gamma_rate"};
  options.set<VariableName>("temperature") = vecstr{"forces", std::string("T")};
  options.set<VariableName>("grain_size") = vecstr{"forces", "grain_size"};
  options.set<VariableName>("stoichiometry") = vecstr{"forces", "stoichiometry"};
  // options.set<VariableName>("fission_rate") = vecstr{"forces", "fission_rate"};
  options.set<bool>("use_AD_first_derivative") = true;
  options.set<bool>("use_AD_second_derivative") = true;
  options.set<std::string>("model_file_name") = "model.pt";
  return options;
}

DanielFlowRate::DanielFlowRate(const OptionSet & options)
  : Model(options),
    // inputs
    // dt(declare_input_variable<Scalar>(options.get<VariableName>("dt"))),
    _temperature(declare_input_variable<Scalar>("temperature")),
    _grain_size(declare_input_variable<Scalar>("grain_size")),
    _stoichiometry(declare_input_variable<Scalar>("stoichiometry")),
    _trial_effective_stress(declare_input_variable<Scalar>("trial_effective_stress")),
    // fission_rate(declare_input_variable<Scalar>(options.get<VariableName>("fission_rate"))),
    // outputs
    _flow_rate(declare_output_variable<Scalar>("flow_rate")),
    _surrogate(std::make_unique<torch::jit::script::Module>(
        torch::jit::load(options.get<std::string>("model_file_name")))),
    _x_mean(declare_buffer<BatchTensor>(
        "x_mean",
        BatchTensor(torch::tensor({1.850000e+03, 5.000000e-05, 5.000000e+07, 2.034279e-03}),
        0))),
    _x_std(declare_buffer<BatchTensor>(
        "x_std",
        BatchTensor(torch::tensor({2.049391e+02, 2.886175e-05, 2.886175e+07, 2.750995e-03}),
        0))),
    _y_mean(declare_buffer<Scalar>("y_mean", Scalar(-1.448253e+01, default_tensor_options()))),
    _y_std(declare_buffer<Scalar>("y_std", Scalar(5.293281e+00, default_tensor_options())))
{
  // 1. We don't need the parameter gradients
  // 2. We need the parameters to have the same options as ours
  _surrogate->to(torch::kFloat64);
  for (auto param : _surrogate->parameters(/*recursive=*/true))
    param.requires_grad_(false);
}

void
DanielFlowRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "I am too lazy to implement second derivatives");
  neml_assert_dbg(!dout_din, "Try AD");

  if (!out)
    return;

  // Grab the trial stress, temperature, grain size, and stoichiometry
  // These are the inputs to the neural network
  const auto & s = _trial_effective_stress;
  const auto & T = _temperature;
  const auto & G = _grain_size;
  const auto & ST = _stoichiometry;

  // Compute Daniel's flow rate
  auto x = BatchTensor(torch::transpose(torch::vstack({T, G, s, ST}), 0, 1), s.batch_dim());
  auto x0 = (x - _x_mean) / _x_std;
  auto gamma0_dot = Scalar(_surrogate->forward({x0}).toTensor().squeeze(), s.batch_dim());
  auto gamma_dot = math::exp(gamma0_dot * _y_std + _y_mean);

  // // irradiation creep
  // auto c3 = _a7 * fission_rate;
  // auto activation_term3 = std::exp(-_q3 / _temperature[_qp]);
  // auto term_3 = c3 * activation_term3;

  _flow_rate = gamma_dot;
}

// void
// DanielFlowRate::to(const torch::TensorOptions & options)
// {
//   Model::to(options);
//   _surrogate->to(options.device());
// }

} // namespace neml2
