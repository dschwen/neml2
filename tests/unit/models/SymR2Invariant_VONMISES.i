[Tensors]
  [foo]
    type = InitializedSymR2
    values = '1 2 3 4 5 6'
    # This tensor reads
    # A = [ 1 6 5
    #       6 2 4
    #       5 4 3 ]
    # VONMISES(A) = sqrt(234) ~= 15.2970585408
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'state/internal/O'
    input_symr2_values = 'foo'
    output_scalar_names = 'state/internal/VM'
    output_scalar_values = '15.2970585408'
    derivatives_abs_tol = 1e-6
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SymR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/VM'
  []
[]
