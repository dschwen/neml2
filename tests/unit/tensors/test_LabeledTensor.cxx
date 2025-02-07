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

#include <catch2/catch.hpp>

#include "neml2/tensors/LabeledTensor.h"

using namespace neml2;

TEST_CASE("operator()", "[LabeledTensor]")
{
  TorchSize nbatch = 10;

  // Setup the Label
  LabeledAxis info1;
  info1.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SymR2>("second");
  info2.setup_layout();

  SECTION("logically 1D LabeledTensor")
  {
    LabeledTensor<1, 1> A(nbatch, {&info1});
    REQUIRE(A("first").sizes() == TorchShape({nbatch, 6}));
    REQUIRE(A("second").sizes() == TorchShape({nbatch, 1}));
    REQUIRE(A("third").sizes() == TorchShape({nbatch, 1}));
  }

  SECTION("logically 2D LabeledTensor")
  {
    LabeledTensor<1, 2> A(nbatch, {&info1, &info2});
    REQUIRE(A("first", "first").sizes() == TorchShape({nbatch, 6, 1}));
    REQUIRE(A("first", "second").sizes() == TorchShape({nbatch, 6, 6}));
    REQUIRE(A("second", "first").sizes() == TorchShape({nbatch, 1, 1}));
    REQUIRE(A("second", "second").sizes() == TorchShape({nbatch, 1, 6}));
    REQUIRE(A("third", "first").sizes() == TorchShape({nbatch, 1, 1}));
    REQUIRE(A("third", "second").sizes() == TorchShape({nbatch, 1, 6}));
  }
}

TEST_CASE("get", "[LabeledTensor]")
{
  TorchSize nbatch = 10;

  // Setup the Label
  LabeledAxis info1;
  info1.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SymR2>("second");
  info2.setup_layout();

  SECTION("logically 1D LabeledTensor")
  {
    LabeledTensor<1, 1> A(nbatch, {&info1});
    REQUIRE(A.get<SymR2>("first").sizes() == TorchShape({nbatch, 6}));
    REQUIRE(A.get<Scalar>("second").sizes() == TorchShape({nbatch, 1}));
    REQUIRE(A.get<Scalar>("third").sizes() == TorchShape({nbatch, 1}));
  }

  SECTION("logically 2D LabeledTensor")
  {
    LabeledTensor<1, 2> A(nbatch, {&info1, &info2});
    REQUIRE(A.get<SymR2>("first", "first").sizes() == TorchShape({nbatch, 6}));
    REQUIRE(A.get<Scalar>("second", "first").sizes() == TorchShape({nbatch, 1}));
    REQUIRE(A.get<SymR2>("second", "second").sizes() == TorchShape({nbatch, 6}));
    REQUIRE(A.get<Scalar>("third", "first").sizes() == TorchShape({nbatch, 1}));
    REQUIRE(A.get<SymR2>("third", "second").sizes() == TorchShape({nbatch, 6}));
  }
}

TEST_CASE("set", "[LabeledTensor]")
{
  TorchSize nbatch = 10;

  // Setup the Label
  LabeledAxis info1;
  info1.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SymR2>("second");
  info2.setup_layout();

  SECTION("logically 1D LabeledTensor")
  {
    LabeledTensor<1, 1> A(nbatch, {&info1});
    A.set(torch::ones({nbatch, 6}), "first");
    REQUIRE(torch::sum(A("first")).item<double>() == Approx(nbatch * 6));
    REQUIRE(torch::sum(A("second")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("third")).item<double>() == Approx(0));
  }

  SECTION("logically 2D LabeledTensor")
  {
    LabeledTensor<1, 2> A(nbatch, {&info1, &info2});
    A.set(torch::ones({nbatch, 1, 6}), "third", "second");
    REQUIRE(torch::sum(A("first", "first")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("first", "second")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("second", "first")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("second", "second")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("third", "first")).item<double>() == Approx(0));
    REQUIRE(torch::sum(A("third", "second")).item<double>() == Approx(nbatch * 6));
  }
}

TEST_CASE("clone", "[LabeledTensor]")
{
  TorchSize nbatch = 10;

  // Setup the Label
  LabeledAxis info1;
  info1.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SymR2>("second");
  info2.setup_layout();

  LabeledTensor<1, 2> A(nbatch, {&info1, &info2});
  auto B = A.clone();

  REQUIRE(A.axis(0) == B.axis(0));
  REQUIRE(A.axis(1) == B.axis(1));
  REQUIRE(torch::allclose(A.tensor(), B.tensor()));

  // Since B is a deep copy, modifying B shouldn't affect A.
  B.set(torch::ones({nbatch, 1, 6}), "third", "second");
  REQUIRE(torch::sum(A("third", "second")).item<double>() == Approx(0));
  REQUIRE(torch::sum(B("third", "second")).item<double>() == Approx(nbatch * 6));
}

TEST_CASE("slice", "[LabeledTensor]")
{
  TorchSize nbatch = 10;

  // Setup the Label
  LabeledAxis info1;
  info1.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.add<LabeledAxis>("sub1").add<LabeledAxis>("sub2");
  info1.subaxis("sub1").add<SymR2>("first").add<Scalar>("second").prefix("sub1");
  info1.subaxis("sub2").add<Scalar>("first").add<Scalar>("second").prefix("sub2");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SymR2>("second");
  info2.setup_layout();

  SECTION("logically 1D LabeledTensor")
  {
    LabeledTensor<1, 1> A(nbatch, {&info1});
    A.set(2.3 * torch::ones({nbatch, 7}), "sub1");
    auto B = A.slice(0, "sub1");
    REQUIRE(torch::sum(B("sub1_first")).item<double>() == Approx(nbatch * 6 * 2.3));
    REQUIRE(torch::sum(B("sub1_second")).item<double>() == Approx(nbatch * 2.3));
  }

  SECTION("logically 2D LabeledTensor")
  {
    LabeledTensor<1, 2> A(nbatch, {&info1, &info2});
    A.set(-1.9 * torch::ones({nbatch, 7, 6}), "sub1", "second");
    auto B = A.slice(0, "sub1");
    REQUIRE(torch::sum(B("sub1_first", "first")).item<double>() == Approx(0));
    REQUIRE(torch::sum(B("sub1_first", "second")).item<double>() == Approx(nbatch * 6 * 6 * -1.9));
    REQUIRE(torch::sum(B("sub1_second", "first")).item<double>() == Approx(0));
    REQUIRE(torch::sum(B("sub1_second", "second")).item<double>() == Approx(nbatch * 1 * 6 * -1.9));
  }
}
