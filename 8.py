from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# Define the constraint functions
def constraint1(x):
    return x[0] - 2*x[1] + 2

def constraint2(x):
    return -x[0] - 2*x[1] + 6

# Initial guess
x0 = [0, 0]

# Define constraints
constraints = [{'type': 'eq', 'fun': constraint1},
               {'type': 'eq', 'fun': constraint2}]

# Use SQP method to minimize the objective function subject to constraints
result = minimize(objective_function, x0, method='SLSQP', constraints=constraints)

# Print the result
print("Minimum found at:")
print("x =", result.x)
print("Minimum value of the objective function:", result.fun)
