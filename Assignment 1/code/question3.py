import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## QUESTION 3.a

# symbolic variables
theta, r, l = sp.symbols('theta r l', real=True)

# Helper function to generate the row for a single wheel
def get_wheel_row(beta, x, y):
    term1 = sp.cos(theta + beta)
    term2 = sp.sin(theta + beta)
    term3 = x * sp.sin(beta) - y * sp.cos(beta)
    return [term1, term2, term3]

# define specific geometry for the 3 wheels 
beta1 = sp.pi / 2
x1, y1 = l, 0

# wheel 2 (add 120 degrees ccw)
beta2 = beta1 + (2 * sp.pi / 3)
x2 = l * sp.cos(2 * sp.pi / 3) 
y2 = l * sp.sin(2 * sp.pi / 3)

# wheel 3 (120 degrees cw)
beta3 = beta1 - (2 * sp.pi / 3)
x3 = l * sp.cos(-2 * sp.pi / 3)
y3 = l * sp.sin(-2 * sp.pi / 3)

# assemble the unsimplified G(theta) matrix (multiplied by 1/r)
G = (1/r) * sp.Matrix([
    get_wheel_row(beta1, x1, y1),
    get_wheel_row(beta2, x2, y2),
    get_wheel_row(beta3, x3, y3)
])

G_simplified = sp.simplify(G)

print("Simplified G(theta) Matrix:")
sp.pprint(G_simplified)

## QUESTION 3.b
# convert SymPy matrix to a callable NumPy function for the ODE solver
G_func = sp.lambdify((theta, r, l), G_simplified, 'numpy')

# define robot parameters in meters
r_val = 0.10
l_val = 0.25
t_span = [0, 30] 
t_eval = np.linspace(t_span[0], t_span[1], 300)

# define the kinematics differential equation
def robot_kinematics(t, q, u):
    # State q = [x, y, theta]
    theta = q[2]
    # Evaluate G(theta) at current heading
    G_mat = G_func(theta, r_val, l_val)
    # Solve for global velocities
    q_dot = np.linalg.inv(G_mat) @ u
    return q_dot

# Helper function to plot results
def plot_results(sol, title):
    plt.figure(figsize=(12, 10))
    
    # x vs Time
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, sol.y[0], 'b-', linewidth=2)
    plt.title(f'{title}: x vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('x Position (m)')
    plt.grid(True)
    
    # y vs Time
    plt.subplot(2, 2, 2)
    plt.plot(sol.t, sol.y[1], 'r-', linewidth=2)
    plt.title(f'{title}: y vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('y Position (m)')
    plt.grid(True)
    
    # theta vs Time
    plt.subplot(2, 2, 3)
    plt.plot(sol.t, sol.y[2], 'g-', linewidth=2)
    plt.title(f'{title}: Theta vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Angle (rad)')
    plt.grid(True)
    
    # Trajectory Top View
    plt.subplot(2, 2, 4)
    plt.plot(sol.y[0], sol.y[1], 'k-', linewidth=2)
    plt.title(f'{title}: Top View Trajectory (x vs y)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal') # Ensures circular paths look circular
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# B.I: Given Rotation Inputs
u_I = np.array([-2.0, 1.0, 1.0])
sol_I = solve_ivp(
    fun = robot_kinematics,
    t_span= t_span,
    t_eval=t_eval,
    y0= [0, 0, 0],
    args=(u_I,)
)
plot_results(sol_I, "Applied Rotation Inputs")

# B.II: Design Control Signals
G_local = G_func(0, r_val, l_val)

# straight line with 60 degree slope --
v_straight = 0.2
q_dot_local_1 = np.array([
    v_straight * np.cos(np.radians(60)), 
    v_straight * np.sin(np.radians(60)), 
    0
])
u_II_1 = G_local @ q_dot_local_1

print("--- Designed Control Signals ---")
print(f"B.II.1 (60 deg line) wheel speeds: u1={u_II_1[0]:.3f}, u2={u_II_1[1]:.3f}, u3={u_II_1[2]:.3f} rad/s")

sol_II_1 = solve_ivp(robot_kinematics, t_span, [0, 0, 0], t_eval=t_eval, args=(u_II_1,))
plot_results(sol_II_1, "60 Degree Line Motion")

# turn in a 2-meter diameter circle --
radius = 1.0
omega = 0.3
q_dot_local_2 = np.array([radius * omega, 0, omega]) # [v_x, v_y, omega]
u_II_2 = G_local @ q_dot_local_2

print(f"B.II.2 (2m dia circle) wheel speeds: u1={u_II_2[0]:.3f}, u2={u_II_2[1]:.3f}, u3={u_II_2[2]:.3f} rad/s")

sol_II_2 = solve_ivp(robot_kinematics, t_span, [0, 0, 0], t_eval=t_eval, args=(u_II_2,))
plot_results(sol_II_2, "2m Diameter Circle Motion")