import sympy as sp
from sympy import solve

a, b, c, d = sp.symbols('a b c d')
w, x, y, z = sp.symbols('w x y z')

matchgate = sp.Matrix([
    [a, 0, 0, b],
    [0, w, x, 0],
    [0, y, z, 0],
    [c, 0, 0, d]
])

A = sp.Matrix([
    [a, b],
    [c, d]
])

B = sp.Matrix([
    [w, x],
    [y, z]
])

unitary_cond_0 = matchgate * sp.conjugate(matchgate.T) - sp.eye(4)
unitary_cond_1 = sp.conjugate(matchgate.T) * matchgate - sp.eye(4)
det_cond = sp.det(A) - sp.det(B)

transpose_cond = A - B.T

constraints = [
    unitary_cond_0,
    unitary_cond_1,
    det_cond,
]

equations = [
    # unitary_cond_0,
    # unitary_cond_1,
    # det_cond,
    transpose_cond
]
solutions = solve(
    equations,
    a, b, c, d, w, x, y, z,
    dict=True
)
print(solutions)
print('\n'*2)

# print the conditions for the solutions
for solution in solutions:
    print("Solution:")
    for key, value in solution.items():
        print(f"{key} = {value}")
    print("Constraints:")
    for equation in constraints:
        print(f"{equation} = {equation.subs(solution)}")
    print('\n'*2)
    


