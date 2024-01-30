nb = 20 # batch size
nt = 100 # time steps

[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 8
    end = 8
    nstep = ${nb}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = ${nt}
  []

  [start_temperature]
    type = LinspaceScalar
    start = 2200
    end = 2200
    nstep = ${nb}
  []
  [end_temperature]
    type = LinspaceScalar
    start = 2200
    end = 2200
    nstep = ${nb}
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = ${nt}
  []

  [start_grain_size]
    type = LinspaceScalar
    start = 5.000e-05
    end = 5.000e-05
    nstep = ${nb}
  []
  [end_grain_size]
    type = LinspaceScalar
    start = 5.000e-05
    end = 5.000e-05
    nstep = ${nb}
  []
  [grain_sizes]
    type = LinspaceScalar
    start = start_grain_size
    end = end_grain_size
    nstep = ${nt}
  []

  [start_stoichiometry]
    type = LinspaceScalar
    start = 1e-05
    end = 1e-05
    nstep = ${nb}
  []
  [end_stoichiometry]
    type = LinspaceScalar
    start = 1e-05
    end = 1e-05
    nstep = ${nb}
  []
  [stoichiometries]
    type = LinspaceScalar
    start = start_stoichiometry
    end = end_stoichiometry
    nstep = ${nt}
  []

  [exx]
    type = FullScalar
    batch_shape = '(${nb})'
    value = 2.5e-4
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nb})'
    value = 1e-4
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nb})'
    value = 1e-4
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = ${nt}
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
  []
[]

[Models]
  ###############################################################################
  # Use the trial state to precalculate invariant flow directions
  # prior to radial return
  ###############################################################################
  [trial_elastic_strain]
    type = ElasticStrain
    plastic_strain = 'old_state/internal/Ep'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    youngs_modulus = 2e11
    poisson_ratio = 0.345
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [flow_direction]
    type = J2FlowDirection
  []
  [trial_state]
    type = ComposedModel
    models = "trial_elastic_strain cauchy_stress mandel_stress flow_direction"
  []
  ###############################################################################
  # The actual radial return:
  # Since the flow directions are invariant, we only need to integrate
  # the consistency parameter.
  ###############################################################################
  [trial_flow_rate]
    type = ScalarStateRate
    state = 'internal/gamma'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/NM'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'internal/Ep'
  []
  [elastic_strain]
    type = ElasticStrain
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [trial_effective_stress]
    type = ComposedModel
    models = "trial_flow_rate
              plastic_strain_rate plastic_strain elastic_strain
              cauchy_stress mandel_stress vonmises"
  []
  [flow_rate]
    type = DanielFlowRate
    model_file_name = 'regression/for_daniel_new/gold/model.pt'
  []
  [integrate_gamma]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'internal/gamma'
  []
  [implicit_rate]
    type = ComposedModel
    models = "trial_effective_stress flow_rate integrate_gamma"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
    additional_outputs = 'state/internal/gamma'
  []
  [model0]
    type = ComposedModel
    models = "trial_state return_map trial_flow_rate
              plastic_strain_rate plastic_strain"
    additional_outputs = 'state/internal/Ep'
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain cauchy_stress'
  []
[]
