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

#include <catch2/catch.hpp>

#include "utils.h"
#include "neml2/models/Model.h"

using namespace neml2;

TEST_CASE("Separate test for Daniel's flow rate")
{
  load_model("regression/for_daniel_new/model.i");
  auto & model = Factory::get_object<Model>("Models", "flow_rate");

  TorchSize nbatch = 10;
  auto in = LabeledVector::empty({nbatch}, {&model.input()});

  // 'Temp (K)', 'Grain (m)', 'Stress (Pa)', 'Stoich'
  const double test[] = {2.200e+03, 5.5000e-05, 1.e+08, 0.01};

  // Specify trial stress
  auto s_accessor = LabeledAxisAccessor({"state", "internal", "s"});
  auto s_min = Scalar::full(0.0);
  auto s_max = Scalar::full(test[2]);
  auto s = Scalar::linspace(s_min, s_max, nbatch);
  in.set(s, s_accessor);

  // Specify temperature {};
  auto T_accessor = LabeledAxisAccessor({"forces", std::string("T")});
  auto T_min = Scalar::full(test[0]);
  auto T_max = Scalar::full(test[0]);
  auto T = Scalar::linspace(T_min, T_max, nbatch);
  in.set(T, T_accessor);

  // Grain size
  auto G_accessor = LabeledAxisAccessor({"forces", "grain_size"});
  auto G_min = Scalar::full(test[1]);
  auto G_max = Scalar::full(test[1]);
  auto G = Scalar::linspace(G_min, G_max, nbatch);
  in.set(G, G_accessor);

  // stoichiometry
  auto ST_accessor = LabeledAxisAccessor({"forces", "stoichiometry"});
  auto ST_min = Scalar::full(test[3]);
  auto ST_max = Scalar::full(test[3]);
  auto ST = Scalar::linspace(ST_min, ST_max, nbatch);
  in.set(ST, ST_accessor);

  // See what parameters the model has
  for (auto && [name, param] : model.named_parameters(/*recurse=*/true))
    std::cout << name << ": " << param.sizes() << std::endl;

  // print_general(model.input(), "input variables");
  // print_general(model.output(), "output variables");

  // compute
  auto out = model.value(in);
  std::cout << out.tensor() << '\n';
  std::cout << s << '\n';

  // expect out: 1.9764e-06
}
