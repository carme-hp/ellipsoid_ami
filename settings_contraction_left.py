import numpy as np
import sys, os
import time

script_path = os.path.dirname(os.path.abspath(__file__))
var_path = os.path.join(script_path, "variables")
sys.path.insert(0, var_path)

import variables

n_ranks = (int)(sys.argv[-1])

# parameters
force = variables.force
scenario_name = variables.scenario_name  

if len(sys.argv) > 3:                                                                           
  scenario_name = sys.argv[0]
  force = float(sys.argv[1])
  print("scenario_name: {}".format(scenario_name))
  print("force: {}".format(force))

if len(sys.argv) > 4:
  individuality_parameter = sys.argv[2] 
else:
  individuality_parameter = str(int(time.time()))

tendon_length_0 = variables.physical_offset[2] + variables.zmin - variables.zmax
tendon_length_t = tendon_length_0
tendon_end_t = variables.physical_offset[2] + variables.zmin
tendon_start_t = variables.zmax

meshes = variables.meshes

# set Dirichlet BC, fix bottom
elasticity_dirichlet_bc_1 = {}

k1 = 0
# fix z value on the whole x-y-plane
for j in range(variables.bs_y):
  for i in range(variables.bs_x):
    elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + j*variables.bs_x + i] = [None,None,0.0,None,None,None]

# fix left edge 
for j in range(variables.bs_y):
  elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + j*variables.bs_x + 0][0] = 0.0
  
# fix front edge 
for i in range(variables.bs_x):
  elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + 0*variables.bs_x + i][1] = 0.0

traction_vector = [0, 0, 0]

elasticity_neumann_bc_1 = [{"element": (variables.el_z-1)*variables.el_x*variables.el_y + i*variables.el_y + j, "constantVector": traction_vector, "face": "2+", "isInReferenceConfiguration": True} for i in range(variables.el_x) for j in range(variables.el_y)]



# callback for result
def handle_result_prestretch(result):
  data = result[0]

  number_of_nodes = variables.bs_x * variables.bs_y
  average_z_start = 0
  average_z_end = 0

  z_data = data["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(variables.bs_z -1) + i]

  average_z_start /= number_of_nodes
  average_z_end /= number_of_nodes

  length_of_muscle = np.abs(average_z_end - average_z_start)
  print("length of muscle (prestretch): ", length_of_muscle)

  if data["timeStepNo"] == 0:
    f = open("muscle_length_prestretch"+individuality_parameter+".csv", "w")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  else:
    f = open("muscle_length_prestretch"+individuality_parameter+".csv", "a")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  
  if data["timeStepNo"] == 1:
    field_variables = data["data"]
    
    strain = max(field_variables[1]["components"][2]["values"])
    stress = max(field_variables[5]["components"][2]["values"])
    
    print("strain: {}, stress: {}".format(strain, stress))
    
    with open("result.csv","a") as f:
      f.write("{},{},{}\n".format(scenario_name,strain,stress))



def callback_function_contraction_1(raw_data):
  global tendon_start_t
  t = raw_data[0]["currentTime"]
  number_of_nodes = variables.bs_x * variables.bs_y
  average_z_start = 0
  average_z_end = 0

  z_data = raw_data[0]["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(variables.bs_z -1) + i]

  average_z_start /= number_of_nodes
  average_z_end /= number_of_nodes
  tendon_start_t = average_z_end

  length_of_muscle = np.abs(average_z_end - average_z_start)
  print("length of muscle 1 (contraction): ", length_of_muscle)

  if t == variables.dt_3D:
    f = open("muscle_length_contraction"+individuality_parameter+"_1.csv", "w")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  else:
    f = open("muscle_length_contraction"+individuality_parameter+"_1.csv", "a")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()


def updateNeumannContraction_1(t):
  tendon_length_t = tendon_end_t - tendon_start_t
  if tendon_length_t > tendon_length_0:
    force = variables.tendon_spring_constant * (tendon_length_t-tendon_length_0)
  else:
    force = 0
  traction_vector = [0,0,force]
  for i in range(variables.el_x):
    for j in range(variables.el_y):
      elasticity_neumann_bc_1[i*variables.el_y+j]["constantVector"] = traction_vector

  config = {
    "InputMeshIsGlobal":  True,
    "divideNeumannBoundaryConditionValuesByTotalArea": True,
    "neumannBoundaryConditions":  elasticity_neumann_bc_1
    }
  return config

config = {
  "scenarioName":                 scenario_name,                # scenario name to identify the simulation runs in the log file
  "logFormat":                    "csv",                        # "csv" or "json", format of the lines in the log file, csv gives smaller files
  "solverStructureDiagramFile":   "solver_structure.txt",       # output file of a diagram that shows data connection between solvers
  "mappingsBetweenMeshesLogFile": "mappings_between_meshes_log.txt",    # log file for mappings 

  "Meshes": meshes,

  "Solvers": {
    "linearElasticitySolver": {           # solver for linear elasticity
      "relativeTolerance":  1e-10,
      "absoluteTolerance":  1e-10,         # 1e-10 absolute tolerance of the residual    ,
      "maxIterations":      1e4,
      "solverType":         "gmres",
      "preconditionerType": "none",
      "dumpFilename":       "",
      "dumpFormat":         "matlab",
    }, 
    "diffusionSolver": {
      "solverType":                     "cg",
      "preconditionerType":             "none",
      "relativeTolerance":              1e-10,
      "absoluteTolerance":              1e-10,
      "maxIterations":                  1e4,
      "dumpFilename":                   "",
      "dumpFormat":                     "matlab"
    },
    "mechanicsSolver": {
      "solverType":                     "preonly",
      "preconditionerType":             "lu",
      "relativeTolerance":              1e-10,
      "absoluteTolerance":              1e-10,
      "maxIterations":                  1e4,
      "snesLineSearchType":             "l2",
      "snesRelativeTolerance":          1e-5,
      "snesAbsoluteTolerance":          1e-5,
      "snesMaxIterations":              10,
      "snesMaxFunctionEvaluations":     1e8,
      "snesRebuildJacobianFrequency":   5,
      "dumpFilename":                   "",
      "dumpFormat":                     "matlab"
    }
  },


  "PreciceAdapter": {        # precice adapter for bottom tendon
      "timeStepOutputInterval":   100,                        # interval in which to display current timestep and time in console
      "timestepWidth":            variables.dt_3D,                          # coupling time step width, must match the value in the precice config
      "couplingEnabled":          True,                       # if the precice coupling is enabled, if not, it simply calls the nested solver, for debugging
      "preciceConfigFilename":    variables.precice_file,    # the preCICE configuration file
      "preciceParticipantName":   "MuscleSolverLeft",             # name of the own precice participant, has to match the name given in the precice xml config file
      "preciceSurfaceMeshes": [                                      # the precice meshes get created as the top or bottom surface of the main geometry mesh of the nested solver
        {
          "preciceMeshName":      "MuscleMeshLeft",         # precice name of the 2D coupling mesh
          "face":                 "2+",                       # face of the 3D mesh where the 2D mesh is located, "2-" = left, "2+" = right (z-coordinate)
        }
      ],
      "preciceSurfaceData": [
        {
          "mode":                 "read-displacements-velocities",    # mode is one of "read-displacements-velocities", "read-traction", "write-displacements-velocities", "write-traction"
          "preciceMeshName":      "MuscleMeshLeft",                 # name of the precice coupling surface mesh, as given in the precice xml settings file
          "displacementsName":    "Displacement",                     # name of the displacements "data", i.e. field variable, as given in the precice xml settings file
          "velocitiesName":       "Velocity",                     # name of the velocities "data", i.e. field variable, as given in the precice xml settings file

        },
        {
          "mode":                 "write-traction",                   # mode is one of "read-displacements-velocities", "read-traction", "write-displacements-velocities", "write-traction"
          "preciceMeshName":      "MuscleMeshLeft",                 # name of the precice coupling surface mesh, as given in the precice xml settings 
          "tractionName":         "Traction",                         # name of the traction "data", i.e. field variable, as given in the precice xml settings file
        }
      ], 
    
    "Coupling": {
      "timeStepWidth":            variables.dt_3D,
      "endTime":                  variables.end_time,
      "logTimeStepWidthAsKey":    "dt_3D",
      "durationLogKey":           "duration_3D",
      "connectedSlotsTerm1To2":   {1:2},  # transfer stress to MuscleContractionSolver gamma
      "connectedSlotsTerm2To1":   None,   # transfer nothing back

      "Term1": { # fibers (FastMonodomainSolver)
        "MultipleInstances": { 
          "ranksAllComputedInstances":    list(range(n_ranks)),
          "nInstances":                   1,

          "instances": [{
            "ranks": [0],

            "StrangSplitting": {
              "timeStepWidth":            variables.dt_splitting,
              "logTimeStepWidthAsKey":    "dt_splitting",
              "durationLogKey":           "duration_splitting",
              "timeStepOutputInterval":   100,
              "connectedSlotsTerm1To2":   None, #{0:0,1:1,2:2,3:3,4:4},
              "connectedSlotsTerm2To1":   None, #{0:0,1:1,2:2,3:3,4:4},

              "Term1": { # reaction term
                "MultipleInstances": {
                  "nInstances":   variables.n_fibers_left,

                  "instances": [{
                    "ranks": [0],

                    "Heun": {
                      "timeStepWidth":            variables.dt_0D,
                      "logTimeStepWidthAsKey":    "dt_0D",
                      "durationLogKey":           "duration_0D",
                      "timeStepOutputInterval":   100,

                      "initialValues":                [],
                      "dirichletBoundaryConditions":  {},
                      "dirichletOutputFilename":      None,
                      "inputMeshIsGlobal":            True,
                      "checkForNanInf":               False,
                      "nAdditionalFieldVariables":    0,
                      "additionalSlotNames":          [],
                      "OutputWriter":                 [],

                      "CellML": {
                        "modelFilename":          variables.input_dir + "hodgkin_huxley-razumova.cellml",
                        "meshName":               "fiber{}_1".format(fiber), 
                        "stimulationLogFilename": "out/" + scenario_name + "stimulation.log",

                        "statesInitialValues":                        [],
                        "initializeStatesToEquilibrium":              False,
                        "initializeStatesToEquilibriumTimeStepWidth": 1e-4,
                        "optimizationType":                           "vc",
                        "approximateExponentialFunction":             True,
                        "compilerFlags":                              "-fPIC -O3 -march=native -Wno-deprecated_declarations -shared",
                        "maximumNumberOfThreads":                     0,

                        "setSpecificStatesCallEnableBegin":       variables.specific_states_call_enable_begin_1,
                        "setSpecificStatesCallFrequency":         variables.specific_states_call_frequency,
                        "setSpecificStatesRepeatAfterFirstCall":  0.01,
                        "setSpecificStatesFrequencyJitter":       [0] ,
                        "setSpecificStatesCallInterval":          0,
                        "setSpecificStatesFunction":              None,
                        "additionalArgument":                     None, 

                        "mappings": {
                          ("parameter", 0):               "membrane/i_Stim",
                          ("parameter", 1):               "Razumova/l_hs",
                          ("parameter", 2):               ("constant", "Razumova/rel_velo"),
                          ("connectorSlot", "vm1"):        "membrane/V",
                          ("connectorSlot", "stress1"):    "Razumova/activestress",
                          ("connectorSlot", "alpha1"):     "Razumova/activation",
                          ("connectorSlot", "lambda1"):    "Razumova/l_hs",
                          ("connectorSlot", "ldot1"):      "Razumova/rel_velo"
                        },
                        "parametersInitialValues": [0.0, 1.0, 0.0],
                      },
                    }
                  } for fiber in range(variables.n_fibers_left)] 
                }
              },

              "Term2": { # diffusion term
                "MultipleInstances": {
                  "nInstances": variables.n_fibers_left,

                  "OutputWriter": [
                    {
                      "format":             "Paraview",
                      "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
                      "filename":           "out/" + scenario_name + "/fibers_1",
                      "fileNumbering":      "incremental",
                      "binary":             True,
                      "fixedFormat":        False,
                      "onlyNodalValues":    True,
                      "combineFiles":       True
                    }
                  ],

                  "instances": [{
                    "ranks": [0],

                    "ImplicitEuler": {
                      "timeStepWidth":            variables.dt_1D,
                      "logTimeStepWidthAsKey":    "dt_1D",
                      "durationLogKey":           "duration_1D",
                      "timeStepOutputInterval":   100,

                      "nAdditionalFieldVariables":    4,
                      "additionalSlotNames":          ["stress1", "alpha1", "lambda1", "ldot1"],

                      "solverName":                       "diffusionSolver",
                      "timeStepWidthRelativeTolerance":   1e-10,

                      "dirichletBoundaryConditions":      {},
                      "dirichletOutputFilename":          None,
                      "inputMeshIsGlobal":                True,
                      "checkForNanInf":                   False,
                      "OutputWriter":                     [],

                      "FiniteElementMethod": {
                        "meshName":           "fiber{}_1".format(fiber),
                        "inputMeshIsGlobal":  True,
                        "solverName":         "diffusionSolver",
                        "prefactor":          variables.diffusion_prefactor,
                        "slotName":           "vm1"
                      }
                    }
                  } for fiber in range(variables.n_fibers_left)]
                }
              }
            }
          }]
        },

        "fiberDistributionFile":                              variables.fiber_distribution_file,
        "firingTimesFile":                                    variables.firing_times_file,
        "valueForStimulatedPoint":                            20.0,
        "onlyComputeIfHasBeenStimulated":                     True,
        "disableComputationWhenStatesAreCloseToEquilibrium":  True,
        "neuromuscularJunctionRelativeSize":                  0.0,################################change for no randomness
        "generateGPUSource":                                  True,
        "useSinglePrecision":                                 False
      },

      "Term2": { # solid mechanics (MuscleContractionSolver)
        "MuscleContractionSolver": {
          "Pmax":                         variables.pmax,
          # "slotNames":                    ["lambdaContraction", "ldotContraction", "gammaContraction", "TContraction"],
          "slotNames":                    ["lambdaContraction1", "ldotContraction1", "gammaContraction1", "TContraction1"],
          "dynamic":                      True,

          "numberTimeSteps":              1,
          "timeStepOutputInterval":       100,
          "lambdaDotScalingFactor":       1,
          "enableForceLengthRelation":    True,
          "mapGeometryToMeshes":          ["fiber{}_1".format(fiber) for fiber in range(variables.n_fibers_left)],

          "OutputWriter": [
            {
              "format":             "Paraview",
              "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
              "filename":           "out/" + scenario_name + "/mechanics_1",
              "fileNumbering":      "incremental",
              "binary":             True,
              "fixedFormat":        False,
              "onlyNodalValues":    True,
              "combineFiles":       True
            }
          ],

          "DynamicHyperelasticitySolver": {
            "durationLogKey":         "duration_3D",
            "logTimeStepWidthAsKey":  "dt_3D",
            "numberTimeSteps":        1,
            "materialParameters":     variables.material_parameters,
            "density":                variables.rho,
            "timeStepOutputInterval": 1,

            "meshName":                   "3Dmesh_quadratic_1",
            "inputMeshIsGlobal":          True,
            "fiberDirection":             [0,0,1],
            "solverName":                 "mechanicsSolver",
            "displacementsScalingFactor":  1.0,
            "useAnalyticJacobian":        True,
            "useNumericJacobian":         False,
            "dumpDenseMatlabVariables":   False,
            "loadFactorGiveUpThreshold":  1,
            "loadFactors":                [],
            "scaleInitialGuess":          False,
            "extrapolateInitialGuess":    True,
            "nNonlinearSolveCalls":       1,

            "dirichletBoundaryConditions":                            elasticity_dirichlet_bc_1,
            "neumannBoundaryConditions":                              None,
            "updateDirichletBoundaryConditionsFunction":              None,
            "updateNeumannBoundaryConditionsFunction":                None,
            "updateDirichletBoundaryConditionsFunctionCallInterval":  1,
            "divideNeumannBoundaryConditionValuesByTotalArea":        True,

            "initialValuesDisplacements": [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
            "initialValuesVelocities":    [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
            "constantBodyForce":          (0, 0, 0),

            "dirichletOutputFilename":    "out/" + scenario_name + "/dirichlet_output_1",
            "residualNormLogFilename":    "out/" + scenario_name + "/residual_norm_log_1.txt",
            "totalForceLogFilename":      "out/" + scenario_name + "/total_force_log_1.txt",

            "OutputWriter": [
              {
                "format": "PythonCallback",
                "callback": variables.muscle_left_write_to_file,
                "outputInterval": 1,
              }
            ],
            "pressure":       { "OutputWriter": [] },
            "dynamic":        { "OutputWriter": [] },
            "LoadIncrements": { "OutputWriter": [] }
          }
        }
      }
    }
  }
    
}
