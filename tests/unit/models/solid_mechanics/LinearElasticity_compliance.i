[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'state/S'
    input_symr2_values = 'S'
    output_symr2_names = 'state/internal/Ee'
    output_symr2_values = 'Ee'
  []
[]

[Tensors]
  [Ee]
    type = InitializedSymR2
    values = '0.09 0.04 -0.02'
  []
  [S]
    type = InitializedSymR2
    values = '13.2692 9.4231 4.8077'
  []
[]

[Models]
  [model]
    type = LinearElasticity
    youngs_modulus = 100
    poisson_ratio = 0.3
    compliance = true
  []
[]
