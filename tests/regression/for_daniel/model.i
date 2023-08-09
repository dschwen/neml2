[Tensors]
  [end_time]
    type = LogSpaceTensor
    start = -3
    end = -3
    steps = 20
  []
  [times]
    type = LinSpaceTensor
    end = end_time
    steps = 100
  []
  [start_temperature]
    type = LinSpaceTensor
    start = 100
    end = 1000
    steps = 20
  []
  [end_temperature]
    type = LinSpaceTensor
    start = 200
    end = 1500
    steps = 20
  []
  [temperatures]
    type = LinSpaceTensor
    start = start_temperature
    end = end_temperature
    steps = 100
  []
  [max_strain]
    type = InitializedSymR2
    values = '0.1 -0.05 -0.05'
    nbatch = 20
  []
  [strains]
    type = LinSpaceTensor
    end = max_strain
    steps = 100
  []
[]

[Drivers]
  [driver]
    type = ThermalStructuralDriver
    model = 'model'
    times = 'times'
    prescribed_temperatures = 'temperatures'
    prescribed_strains = 'strains'
    verbose = true
  []
[]

[Solvers]
  [newton]
    type = NewtonNonlinearSolver
    verbose = true
    abs_tol = 1e-8
    rel_tol = 1e-6
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [flow_rate]
    type = DanielFlowRate
    parameter_1 = 100
    parameter_2 = 0.1
    parameter_3 = 2
  []
  [flow_direction]
    type = J2FlowDirection
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SymR2ForceRate
    force = 'E'
  []
  [Eerate]
    type = ElasticStrain
    rate_form = true
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
  []
  [integrate_stress]
    type = SymR2BackwardEulerTimeIntegration
    variable = 'S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress flow_rate flow_direction Eprate Erate Eerate elasticity integrate_stress'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
