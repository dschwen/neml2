[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    nbatch = 10
    input_symr2_names = 'forces/E_rate state/internal/Ep_rate'
    input_symr2_values = 'E_rate Ep_rate'
    output_symr2_names = 'state/internal/Ee_rate'
    output_symr2_values = 'Ee_rate'
  []
[]

[Tensors]
  [E_rate]
    type = InitializedSymR2
    values = '0.100 0.110 0.100 0.050 0.040 0.030'
  []
  [Ep_rate]
    type = InitializedSymR2
    values = '0.050 -0.010 0.020 0.040 0.030 -0.060'
  []
  [Ee_rate]
    type = InitializedSymR2
    values = '0.050 0.120 0.080 0.010 0.010 0.090'
  []
[]

[Models]
  [model]
    type = ElasticStrain
    rate_form = true
  []
[]
