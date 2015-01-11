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
                           field_names=['w','u','p','row'],
                           param_names=['N','g'])

#intial equation
problem.add_equation("dx(u) + dz(w)=0")
problem.add_equation("dt(row) + w*N = 0")
problem.add_equation("row*dt(u) = - dx(p)")
problem.add_equation("row*dt(w) +  g*row = -dz(p)")

#auxillary equations

#defines bc
problem.add_left_bc("w=0");
problem.add_left_bc("u=0");
problem.add_right_bc("u=0");
problem.add_right_bc("w=0");



problem.parameters['N'] = 1
problem.parameters['g'] = 9.8

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
w['g'] = np.cos(3.14*x + 1.57*z)

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
    plt.savefig('frames_2/foo_'+str(i)+"_.png")






