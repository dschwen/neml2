nbatch=5
[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -3
    end = -3
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []

  [start_temperature]
    type = LinspaceScalar
    start = 1970
    end = 1980
    nstep = ${nbatch}
  []
  [end_temperature]
    type = LinspaceScalar
    start = 2000
    end = 2010
    nstep = ${nbatch}
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = 100
  []

  [start_grain_size]
    type = LinspaceScalar
    start = 5.9000e-05
    end = 6.000e-05
    nstep = ${nbatch}
  []
  [end_grain_size]
    type = LinspaceScalar
    start = 8.9000e-05
    end = 9.000e-05
    nstep = ${nbatch}
  []
  [grain_sizes]
    type = LinspaceScalar
    start = start_grain_size
    end = end_grain_size
    nstep = 100
  []


  [start_stoichiometry]
    type = LinspaceScalar
    start = 6.9688e-05
    end = 7.000e-05
    nstep = ${nbatch}
  []
  [end_stoichiometry]
    type = LinspaceScalar
    start = 8.5000e-05
    end = 9.000e-05
    nstep = ${nbatch}
  []
  [stoichiometries]
    type = LinspaceScalar
    start = start_stoichiometry
    end = end_stoichiometry
    nstep = 100
  []

  [exx]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 2.9e+07
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = ThermalStructuralDriver
    model = 'model'
    times = 'times'
    prescribed_temperatures = 'temperatures'
    prescribed_strains = 'strains'
    prescribed_grain_sizes = 'grain_sizes'
    prescribed_stoichiometries = 'stoichiometries'
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
  []
  [flow_direction]
    type = J2FlowDirection
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2ForceRate
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
    type = SR2BackwardEulerTimeIntegration
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
