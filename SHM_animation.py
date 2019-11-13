import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Mass-spring damper system parameters
nu = 5e-2                   # loss factor
fd = 50                     # drive freq
fn = 50                     # natural freq
omega_d = 2 * np.pi * fd    # drive freq
omega_n = 2 * np.pi * fn    # natural frequency
F0 = 10                     # forcing term
params = [nu, omega_d, omega_n, F0]

# Initial conditions
y0 = 0.0  # no initial displacement
w0 = 0.0  # no initial velocity
yint = [y0, w0]


# Mass-spring damper system's equations of motion
def mass_spring_damper(y, t, params):
    y, w = y  # instantaneous values of y and y_dot
    nu, omega_d, omega_n, F0 = params  # system parameters
    derivs = [w, F0 * np.cos(
        omega_d * t) - omega_n * nu * w - y * omega_n * omega_n]  # derivatives of y and y_dot (two coupled ODEs)
    return derivs


# Initialise time step array
t = np.arange(0, 20/fd, 0.02/fd)

# Call the ODE solver that returns displacement and velocity for each time step
psoln = scipy.integrate.odeint(mass_spring_damper, yint, t, args=(params,))

# Properties for text box displayed on animated graph
props = dict(boxstyle='round', facecolor='w', alpha=1)
textstr = '\n'.join((
    r'$f_{n}=%.1f$ Hz' % fn,
    r'$f_{d}=%.1f$ Hz' % fd,
    r'$\nu=%.2f$' % nu))


# Animate function
def animate(i):
    plt.cla()
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.set_xlabel('Time(s)', size=12)
    ax.set_ylabel('Velocity (m/s)', size=12)
    ax.set_title('Simple Harmonic Motion of a 1-DOF Oscillator', pad=10, size=14)
    ln1 = ax.plot(t[:i], psoln[:i,1], color='b', label='Velocity', lw=1, linestyle='-')
    ln2 = ax.plot(t[i-1], psoln[i-1,1], marker='o', markersize=6, color="red")


# Animate plot of system's velocity (single degree-of-freedom system)
fig, ax = plt.subplots(1, 1, figsize=(9, 8))
ani = FuncAnimation(fig, animate, #init_func=init,
                    interval=20,
                    frames=len(t),
                    repeat=False)
plt.show()