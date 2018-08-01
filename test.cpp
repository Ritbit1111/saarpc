import sympy
from sympy.physics.vector import *
from sympy import Curve, line_integrate, E, ln, diff
import numpy as np
import matplotlib.pyplot as plt

# Establish coordinates for calculus
R = ReferenceFrame('R')
x, y = R[0], R[1]

# Establish coordinates for plots
X,Y = np.meshgrid(np.arange(-10,11), np.arange(-10,11))

# ~~~~~~~~~~~~~~~~~~~~~
# LOTS OF BORING STUFF:
# ~~~~~~~~~~~~~~~~~~~~~

# Turn a 2-D vector to a tuple
def vector_components(vectorField):
    return (vectorField.dot(R.x), vectorField.dot(R.y))

# Apply a field to a discrete set of points X, Y
def discretize_field(field):
    computeVector = sympy.lambdify((x,y), field)
    return computeVector(X,Y)
    
def plot_streamlines(vectorField):
    data = discretize_field(vector_components(vectorField))
    plt.figure()
    plt.streamplot(X, Y, data[0], data[1], density=2, linewidth=1, arrowsize=2, arrowstyle='->')

# Return a scalar potential function of a vector field
def integrateGradient(gradientField):
    return sympy.integrate(gradientField.dot(R.x), x) + sympy.integrate(gradientField.dot(R.y), y).subs(x,0)

# ~~~~~~~~~~~~~~~~~~~~~
# THE FUN STARTS HERE!
# ~~~~~~~~~~~~~~~~~~~~~

# The given vector field. (Read R.x and R.y like the unit vectors i-hat and j-hat.)
u, v = (-2*y * (1-x**2)), (2*x * (1-y**2))
velocity = u*R.x + v*R.y

# Plot the vector field:
plot_streamlines(velocity)

# Find the stagnation points.
# The solve(v) function gives solutions for v = 0
stagPoints = sympy.solve(vector_components(velocity), [x,y]);
for (x_point,y_point) in stagPoints:
    plt.scatter(x_point, y_point, color='#CD2305', s=80, marker='o')

# Does the field satisfy conservation of mass for an incompressible flow?
print "The divergence of the field is", divergence(velocity, R), "."

# Is it a potential flow?
print "The vorticity of the field is", curl(velocity, R), "."

# Find the stream function
u, v = velocity.dot(R.x), velocity.dot(R.y)
streamField = u * R.y - v * R.x
print "Stream function", integrateGradient(streamField)

plt.show()

        
        
        
        
        ########################################new#######################################
        import numpy as np
import matplotlib.pyplot as plt
    
#length of the plates of the capacitor
x1,y1 = 3,3
x2,y2 = 3,5
lx=4
ly=4
#The capacitor is placed on a bigger box of size N1xN2
Nx=6*lx
Ny=6*ly
x=np.linspace(0,Nx-1,num=Nx)
y=np.linspace(0,Ny-1,num=Ny)

X,Y = np.meshgrid(x,y)

#Potential is 2D variable
V=np.zeros((Nx,Ny))

V[x1:x1+lx,y1]=6.5
V[x2,y2:y2+ly]=6.5
#print(V)
#Potential at each point is the average of its neighbor barring the boundary codidtions
iter=1000

count=0
while count<iter:
    for i in range(1,Nx-2):
        for j in range(1,Ny-2):
            if i>=x1 and i<x1+lx and j==y1:
                #print(i,j,V[i][j])
                continue
            if j>=y2 and j<y2+ly and i==x2:
                #print(i,j,V[i][j])
                continue
            V[i][j]=(V[i-1][j]+V[i+1][j]+V[i][j-1]+V[i][j+1])/4
    count=count+1
print(V)

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(V)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()
