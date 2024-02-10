import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Define the convex univariate function
def convex_function(x):
    return (x - 2) ** 2

# Optimize the function
optimum = minimize_scalar(convex_function).x

# Generate data for plotting
x_values = np.linspace(-2, 8, 100)
y_values = convex_function(x_values)

# Plot the function and mark the optimum
plt.plot(x_values, y_values, label='Convex Function')
plt.scatter(optimum, convex_function(optimum), color='red', label='Optimum')

# Annotate the optimum point
plt.annotate('Optimum', xy=(optimum, convex_function(optimum)), xytext=(optimum+1 , convex_function(optimum)+5), arrowprops=dict(facecolor="Black", shrink=0.2))

# Set plot labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Optimization of Convex Univariate Function')
plt.legend()
plt.grid()
plt.show()
