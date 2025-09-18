import sys
import os
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ellipsoid_mesh_generation

if sys.argv[-1] != "BayesOpt.py":
    n_ranks = int(sys.argv[-1])
    rank_no = int(sys.argv[-2])
else:
    n_ranks = 1


precice_file = "../precice-config.xml"
# Time stepping
dt_3D = 1            # time step of 3D mechanics
dt_splitting = 2e-3     # time step of strang splitting
dt_1D = 2e-3            # time step of 1D fiber diffusion
dt_0D = 1e-3            # time step of 0D cellml problem
end_time = 100.0         # end time of the simulation 
output_interval = dt_3D # time interval between outputs

# Material parameters
pmax = 7.3                                                  # maximum active stress
rho = 10                                                    # density of the muscle
material_parameters = [3.176e-10, 1.813, 1.075e-2, 1.0]     # [c1, c2, b, d]
diffusion_prefactor = 3.828 / (500.0 * 0.58)                # Conductivity / (Am * Cm)

force = 10.0
scenario_name = "incompressible_mooney_rivlin"

# 3D Meshes
el_x, el_y, el_z = 2, 2, 6                     # number of elements
bs_x, bs_y, bs_z = 2*el_x+1, 2*el_y+1, 2*el_z+1 # quadratic basis functions

c = 4.5 # perpendicular to fiber direction z
a = 8.8 # fiber direction z
zmin = -6.6
zmax = 7.1

physical_offset = [0, 0, 16.8]

nodes_tendon = ellipsoid_mesh_generation.mesh_nodes(el_x*2,el_y*2, el_z*2,2.8,70,zmax,10.33333)
nodes_left = ellipsoid_mesh_generation.mesh_nodes(el_x*2,el_y*2, el_z*2, c, a, zmin, zmax)
nodes_right = ellipsoid_mesh_generation.apply_offset(nodes_left,physical_offset)
print(nodes_left[0])
print(nodes_right[0])

meshes = { # create 3D mechanics mesh
    "3Dmesh_quadratic_1": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [el_x, el_y, el_z],               # number of quadratic elements in x, y and z direction
      "nodePositions":              nodes_left,      
    },
	"3Dmesh_tendon": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [el_x, el_y, el_z],               # number of quadratic elements in x, y and z direction
      "nodePositions":              nodes_tendon,      
    },
    "3Dmesh_quadratic_2": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [el_x, el_y, el_z],               # number of quadratic elements in x, y and z direction
      "nodePositions":              nodes_right,    
    }
}

# n 1D meshes left

with open("../mesh_left.json","r") as f:
	fdata_left = json.load(f)
      
fb_points_left = 99           # number of points per fiber

fiber_idx = 0
for fiber in fdata_left:
	fdict = fdata_left[fiber]
	npos = [[fdict[ii]['x'],fdict[ii]['y'],fdict[ii]['z']] for ii in range(len(fdict)-1) ]
	meshName = "fiber{}_1".format(fiber_idx)
	meshes[meshName] = {
			"nElements":		    [len(npos)-1],
			"nodePositions":	    npos,
			"inputMeshIsGlobal":	True,
			"nRanks":				n_ranks
	}
	fiber_idx += 1
     
n_fibers_left = fiber_idx

# n 1D meshes right

with open("../mesh_right.json","r") as f:
	fdata_right = json.load(f)
      
fb_points_right = 99           # number of points per fiber

fiber_idx = 0
for fiber in fdata_right:
	fdict = fdata_right[fiber]
	npos = [[fdict[ii]['x'],fdict[ii]['y'],fdict[ii]['z']] for ii in range(len(fdict)-1) ]
	meshName = "fiber{}_2".format(fiber_idx)
	meshes[meshName] = {
			"nElements":		    [len(npos)-1],
			"nodePositions":	    npos,
			"inputMeshIsGlobal":	True,
			"nRanks":				n_ranks
	}
	fiber_idx += 1
     
n_fibers_right = fiber_idx


# Define directory for cellml files
input_dir = os.path.join(os.environ.get('OPENDIHU_HOME', '../../../../../'), "examples/electrophysiology/input/")

# Fiber activation
fiber_distribution_file = input_dir + "MU_fibre_distribution_3780.txt"
firing_times_file = input_dir + "MU_firing_times_always.txt"
specific_states_call_enable_begin_2 = 1.0                  # time of first fiber activation
specific_states_call_enable_begin_1 = end_time  
specific_states_call_frequency = 1e-5                       # frequency of fiber activation

tendon_spring_constant = 10

constant_body_force = (0,0,0)

#### added for the tendon
n_subdomains_x = 1
n_subdomains_y = 1
n_subdomains_z = 1

tendon_material = "nonLinear"
#tendon_material = "SaintVenantKirchoff"         

tendon_extent = [3.0, 3.0, 5.0] # [cm, cm, cm] 2.96
n_elements_tendon = [8,8,8] 
tendon_offset = [0.0, 0.0, 5.0]

elasticity_dirichlet_bc = {}

def get_from_obj(data, path):
    for elem in path:
        if type(elem) == str:
            data = data[elem]
        elif type(elem) == int:
            data = data[elem]
        elif type(elem) == tuple:
            # search for key == value with (key, value) = elem
            key, value = elem
            data = next(filter(lambda e: e[key] == value, data))
        else:
            raise KeyError(f"Unknown type of '{elem}': '{type(elem)}'. Path: '{'.'.join(path)}'")
    return data

def tendon_write_to_file(data):
    t = get_from_obj(data, [0, 'currentTime'])
    z_data = get_from_obj(data, [0, 'data', ('name','geometry'), 'components', 2, 'values'])

    [mx, my, mz] = get_from_obj(data, [0, 'nElementsLocal'])
    nx = 2*mx + 1
    ny = 2*my + 1
    nz = 2*mz + 1
    # compute average z-value of end of muscle
    z_value_begin = 0.0
    z_value_end = 0.0

    for j in range(ny):
        for i in range(nx):
            z_value_begin += z_data[j*nx + i]
            z_value_end += z_data[(nz-1)*nx*ny + j*nx + i]

    z_value_begin /= ny*nx
    z_value_end /= ny*nx


    f = open( "tendon.txt", "a")
    f.write("{:6.2f} {:+2.8f} {:+2.8f}\n".format(t,z_value_begin, z_value_end))
    f.close()

def muscle_left_write_to_file(data):
    t = get_from_obj(data, [0, 'currentTime'])
    z_data = get_from_obj(data, [0, 'data', ('name','geometry'), 'components', 2, 'values'])

    [mx, my, mz] = get_from_obj(data, [0, 'nElementsLocal'])
    nx = 2*mx + 1
    ny = 2*my + 1
    nz = 2*mz + 1
    # compute average z-value of end of muscle
    z_value_begin = 0.0
    z_value_end = 0.0

    for j in range(ny):
        for i in range(nx):
            z_value_begin += z_data[j*nx + i]
            z_value_end += z_data[(nz-1)*nx*ny + j*nx + i]


    z_value_begin /= ny*nx
    z_value_end /= ny*nx


    f = open("muscle_left.txt", "a")
    f.write("{:6.2f} {:+2.8f} {:+2.8f}\n".format(t,z_value_begin, z_value_end))
    f.close()

def muscle_right_write_to_file(data):
    t = get_from_obj(data, [0, 'currentTime'])
    z_data = get_from_obj(data, [0, 'data', ('name','geometry'), 'components', 2, 'values'])

    [mx, my, mz] = get_from_obj(data, [0, 'nElementsLocal'])
    nx = 2*mx + 1
    ny = 2*my + 1
    nz = 2*mz + 1
    # compute average z-value of end of muscle
    z_value_begin = 0.0
    z_value_end = 0.0

    for j in range(ny):
        for i in range(nx):
            z_value_begin += z_data[j*nx + i]
            z_value_end += z_data[(nz-1)*nx*ny + j*nx + i]

    z_value_begin /= ny*nx
    z_value_end /= ny*nx

    f = open("muscle_right.txt", "a")
    f.write("{:6.2f} {:+2.8f} {:+2.8f}\n".format(t,z_value_begin, z_value_end))
    f.close()