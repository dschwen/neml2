[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_scalar_names = 'state/internal/k'
    input_scalar_values = '20'
    input_symr2_names = 'state/internal/M state/internal/X'
    input_symr2_values = 'M X'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '83.5577'
    check_second_derivatives = true
    derivatives_abs_tol = 1e-06
  []
[]

[Tensors]
  [M]
    type = InitializedSymR2
    values = '100 110 100 50 40 30'
  []
  [X]
    type = InitializedSymR2
    values = '60 -10 20 40 30 -60'
  []
[]

[Models]
  [overstress]
    type = OverStress
  []
  [vonmises]
    type = SymR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/sm'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    isotropic_hardening = 'state/internal/k'
  []
  [model]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
[]
