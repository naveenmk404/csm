import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')
f = 1 / x

slope_sym = sp.diff(f, x)

x_values = np.linspace(-3, 3, 400)
y_values = [f.evalf(subs={x: val}) for val in x_values]

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='f(x) = 1/x')

for x_val in [1, -1]:
    slope = slope_sym.evalf(subs={x: x_val})
    tangent_line = slope * (x_values - x_val) + f.evalf(subs={x: x_val})
    plt.plot(x_values, tangent_line, label=f'Tangent at x={x_val}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = 1/x and Tangent Lines')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()
