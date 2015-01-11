from dedalus2 import public as de
from dedalus2.extras import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import itertools

def matrix_plot(matrix, clim):
    if np.isscalar(clim):
        clim = (-clim, clim)
    plt.imshow(matrix, clim=clim, cmap='RdBu_r',interpolation='none')
    plt.colorbar()
    plt.savefig("basis")
    
def sparse_matrix_plot(matrix, clim):
    matrix_plot(matrix.todense(), clim)



#
# DEFINES INTIAL EQUATION
#

problem = de.ParsedProblem(axis_names=['x','z'],
                           field_names=['w','wt',
                                        'wzt','wzzt'],
                           param_names=['N'])

#intial equation
problem.add_equation("dt(wzzt) + dx(dx(dt(wt))) + N*dx(dx(w)) = 0 ")
problem.add_equation("dt(w) - wt = 0")

#auxillary equations

#defines first order derivates (space)
#problem.add_equation("wx - dx(w) = 0")
problem.add_equation("wzt  - dz(wt) =0")

#defines third order derivatives (space + time)
problem.add_equation("dz(wzt) - wzzt = 0")

#defines bc
problem.add_left_bc("w=0");
problem.add_right_bc("w=0");
#problem.add_int_bc("w=cos(x)");


problem.parameters['N'] = 1

print("made it past problem setup")

#
# DEFINES SOLVER + BASIS
#


x_basis = de.Fourier(128, interval=(0, 1), dealias=2/3)
z_basis = de.Chebyshev(128,interval=(-2,2),dealias=2/3)
domain = de.Domain([x_basis,z_basis], grid_dtype = np.float64)
problem.expand(domain)


solver = de.solvers.IVP(problem, domain, de.timesteppers.SBDF3)

solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 1000

x = domain.grid(0) 
z = domain.grid(1)
w = solver.state['w']

#
#DEFINES INTIAL CONITIONS
#

#w['g'] = np.exp(-(pow((x-.5),2) + pow((z-.5),2)))*np.cos(1.57*x + 1.57*z)
w['g'] = np.cos(1.57*x + 1.57*z)

# Setup storage
w_list = [np.copy(w['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 1e-2
while solver.ok:
    solver.step(dt)
    if solver.iteration % 20 == 0:
        w_list.append(np.copy(w['g']))
        t_list.append(solver.sim_time)



# Convert storage to arrays
w_array = np.array(w_list)
t_array = np.array(t_list)


#NEW PLOTTING

import os
if not os.path.exists('frames'):
   os.mkdir('frames')
xmesh, ymesh = plot_tools.quad_mesh(x=x_basis.grid, y=z_basis.grid)
plt.figure(figsize=(10, 6))
plt.axis(plot_tools.pad_limits(xmesh, ymesh))
plt.xlabel('x')
plt.ylabel('z')
plt.title('internal wave')
for i in range(0,50):
    if i == 0:
        p = plt.pcolormesh(xmesh, ymesh, w_array[i].T, cmap='RdBu_r')
        plt.colorbar()
    else:
        p.set_array(w_array[i].T.ravel())
    plt.savefig('frames/foo_'+str(i)+"_.png")




#OLD PLOTTING
#File handling
#
#f = open('w_data.txt','w')
#for i in range (0, len(x)):
#    np.savetxt("w_data",x[i])

#np.savetxt("x_data.txt",x);
#np.savetxt("t_data.txt",t_array);

# Plot
#xmesh, ymesh = plot_tools.quad_mesh(x=x_basis.grid, y=z_basis.grid)
#plt.figure(figsize=(10,10))
#plt.axis(plot_tools.pad_limits(xmesh, ymesh))
#plt.xlabel('x')
#plt.ylabel('z')
#plt.title('internal wave')
#for i in range(0,50):
#   plt.pcolormesh(xmesh, ymesh, w_array[i], cmap='RdBu_r')
#   if(i==0):
#         plt.colorbar()
#   plt.savefig('foo_'+str(i)+"_.png")


