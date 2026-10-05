import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# define ODE function for solve_ivp
def diff_drive_model(t, state, v_func, omega_func):
    x, y, theta = state
    v = v_func(t)
    omega = omega_func(t)

    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = omega

    return [dxdt, dydt, dthetadt]

# function that solves ODE and outputs the plotted trajectory
# and state variables over time
def run_and_plot_ivp(v_func, omega_func, title):
    # define time boundary (0 -> 30 sec)
    t_span = (0.0, 30.0)
    t_points = np.linspace(t_span[0], t_span[1], 100)

    # initial state [x=0, y=0, theta=0]
    initial_state = [0.0, 0.0, 0.0]

    solution = solve_ivp(
        fun = diff_drive_model,
        t_span = t_span,
        t_eval = t_points,
        y0 = initial_state,
        args = (v_func, omega_func),
    )

    t_arr = solution.t
    x_arr = solution.y[0]
    y_arr = solution.y[1]
    theta_arr = solution.y[2]

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 5))
    
    # Plot 1: Top View (2D Trajectory: x vs y)[cite: 2]
    axs[0,0].plot(x_arr, y_arr, label='Trajectory', color='b', linewidth=2)
    axs[0,0].set_title(f"2D Trajectory ($x$ vs $y$) for {title}")
    axs[0,0].set_xlabel("X Position [m]")
    axs[0,0].set_ylabel("Y Position [m]")
    axs[0,0].grid(True)
    axs[0,0].axis('equal')
    axs[0,0].legend()
    
    # Plot 2: X Position vs Time
    axs[0,1].plot(t_arr, x_arr, label='$x(t)$', color='r', linewidth=1.5)
    axs[0,1].set_title(f"X Position vs Time for {title}")
    axs[0,1].set_xlabel("Time [s]")
    axs[0,1].set_ylabel("x [m]")
    axs[0,1].grid(True)
    axs[0,1].legend()
    
    # Plot 3: Y Position vs Time
    axs[1,0].plot(t_arr, y_arr, label='$y(t)$', color='g', linewidth=1.5)
    axs[1,0].set_title(f"Y Position vs Time for {title}")
    axs[1,0].set_xlabel("Time [s]")
    axs[1,0].set_ylabel("y [m]")
    axs[1,0].grid(True)
    axs[1,0].legend()
    
    # Plot 4: Orientation vs Time
    axs[1,1].plot(t_arr, theta_arr, label=r'$\theta(t)$', color='purple', linewidth=1.5)
    axs[1,1].set_title(f"Orientation vs Time for {title}")
    axs[1,1].set_xlabel("Time [s]")
    axs[1,1].set_ylabel(r'$\theta$ [rad]')
    axs[1,1].grid(True)
    axs[1,1].legend()
    
    plt.tight_layout()
    plt.show()

# Case i.1:
run_and_plot_ivp(lambda t: 1.0, lambda t: 0.0, "Constant v=1, omega=0")

# Case i.2:
run_and_plot_ivp(lambda t: 0.0, lambda t: 0.5, "Constant v=0, omega=0.5")

# Case i.3:
run_and_plot_ivp(lambda t: 1.0, lambda t: 0.5, "Constant v=1, omega=0.5")

# Case ii: Time-varying velocity profiles
v_profile = lambda t: 1.0 + 0.2 * np.sin(t)
omega_profile = lambda t: 0.2 + 0.6 * np.cos(t)
run_and_plot_ivp(v_profile, omega_profile, "Time-Varying Profile")