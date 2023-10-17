"""
Python module models.py for various vehicle models.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

import numpy as np
from mobotpy import graphics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import chi2
from matplotlib import patches


class Cart:
    """1D vehicle class (i.e., a simple cart).

    Parameters
    ----------
    length : float
        Length of the cart [m].
    """

    def __init__(self, length):
        """Constructor method."""
        self.d = length

    def draw(self, x):
        """Finds the points to draw simple rectangular cart.

        The cart has position x and length d.  The resulting cart has a height
        that is half the length.
        """
        X = np.array(5)
        Y = np.array(5)
        X = [
            x - self.d / 2,
            x - self.d / 2,
            x + self.d / 2,
            x + self.d / 2,
            x - self.d / 2,
        ]
        Y = [-self.d / 4, self.d / 4, self.d / 4, -self.d / 4, -self.d / 4]
        return X, Y

    def animate(self, x, T, save_ani=False, filename="animate_cart.gif"):
        """Create an animation of a simple 1D cart.

        Returns animation object for array of 1D cart positions x with time
        increments T [s], cart width d [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default filename is 'animate_cart.gif').
        """
        fig, ax = plt.subplots()
        plt.plot([np.min(x) - self.d, np.max(x) + self.d], [0, 0], "k--")
        plt.xlabel(r"$x$ [m]")
        ax.set_xlim([np.min(x) - self.d, np.max(x) + self.d])
        plt.yticks([])
        plt.axis("equal")
        (polygon,) = ax.fill([], [], "C0", alpha=0.5)
        (line,) = plt.plot([], [], "ko")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        # Initialization function

        def init():
            polygon.set_xy(np.empty([5, 2]))
            line.set_data([], [])
            time_text.set_text("")
            return polygon, line, time_text

        # Function to draw cart

        def movie(k):
            X, Y = self.draw(x[k])
            a = [X, Y]
            polygon.set_xy(np.transpose(a))
            line.set_data(x[k], 0)
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            return polygon, line, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        # Save to a file if requested
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani


class DiffDrive:
    """Differential-drive vehicle class.

    Parameters
    ----------
    ell : float
        The track length of the vehicle [m].
    """

    def __init__(self, ell):
        """Constructor method."""
        self.ell = ell

    def f(self, x, u):
        """Differential drive kinematic vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 3
            The vehicle's state (x, y, theta).
        u : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).

        Returns
        -------
        f : ndarray of length 3
            The rate of change of the vehicle states.
        """
        f = np.zeros(3)
        f[0] = 0.5 * (u[0] + u[1]) * np.cos(x[2])
        f[1] = 0.5 * (u[0] + u[1]) * np.sin(x[2])
        f[2] = 1.0 / self.ell * (u[1] - u[0])
        return f

    def uni2diff(self, u_in):
        """
        Convert speed and anular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        """
        v = u_in[0]
        omega = u_in[1]
        v_L = v - self.ell / 2 * omega
        v_R = v + self.ell / 2 * omega
        u_out = np.array([v_L, v_R])
        return u_out

    def draw(self, x, y, theta):
        """
        Finds points that draw a differential drive vehicle.

        The centre of the wheel axle is (x, y), the vehicle has orientation
        theta, and the vehicle's track length is ell.

        Returns X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C, where L is for the
        left wheel, R for the right wheel, B for the body, and C for the caster.
        """
        # Left and right wheels
        X_L, Y_L = graphics.draw_rectangle(
            x - 0.5 * self.ell * np.sin(theta),
            y + 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        X_R, Y_R = graphics.draw_rectangle(
            x + 0.5 * self.ell * np.sin(theta),
            y - 0.5 * self.ell * np.cos(theta),
            0.5 * self.ell,
            0.25 * self.ell,
            theta,
        )
        # Body
        X_BD, Y_BD = graphics.draw_circle(x, y, self.ell)
        # Caster
        X_C, Y_C = graphics.draw_circle(
            x + 0.5 * self.ell * np.cos(theta),
            y + 0.5 * self.ell * np.sin(theta),
            0.125 * self.ell,
        )
        # Return the arrays of points
        return X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C

    def animate(self, x, T, save_ani=False, filename="animate_diffdrive.gif"):
        """Create an animation of a differential drive vehicle.

        Returns animation object for array of vehicle positions x with time
        increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, leftwheel, rightwheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani

    def animate_trajectory(
        self, x, xd, T, save_ani=False, filename="animate_diffdrive.gif"
    ):
        """Create an animation of a differential drive vehicle with plots of
        actual and desired trajectories.

        Returns animation object for array of vehicle positions and desired
        positions x with time increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (desired,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            desired.set_data([], [])
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return desired, line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            desired.set_data(xd[0, 0: k + 1], xd[1, 0: k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return desired, line, leftwheel, rightwheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani

    def animate_estimation(
        self,
        x,
        x_hat,
        P_hat,
        alpha,
        T,
        save_ani=False,
        filename="animate_diffdrive.gif",
    ):
        """Create an animation of a differential drive vehicle with plots of
        estimation uncertainty.

        Returns animation object for array of vehicle positions x with time
        increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (estimated,) = ax.plot([], [], "--C1")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        (castor,) = ax.fill([], [], color="k")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        s2 = chi2.isf(alpha, 2)

        def init():
            """Function that initializes the animation."""
            estimated.set_data([], [])
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([36, 2]))
            castor.set_xy(np.empty([36, 2]))
            time_text.set_text("")
            return estimated, line, leftwheel, rightwheel, body, castor, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the desired trajectory
            estimated.set_data(x_hat[0, 0: k + 1], x_hat[1, 0: k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Compute eigenvalues and eigenvectors to find axes for covariance ellipse
            W, V = np.linalg.eig(P_hat[0:2, 0:2, k])
            # Find the index of the largest and smallest eigenvalues
            j_max = np.argmax(W)
            j_min = np.argmin(W)
            ell = patches.Ellipse(
                (x_hat[0, k], x_hat[1, k]),
                2 * np.sqrt(s2 * W[j_max]),
                2 * np.sqrt(s2 * W[j_min]),
                angle=np.arctan2(V[j_max, 1], V[j_max, 0]) * 180 / np.pi,
                alpha=0.2,
                color="C1",
            )
            ax.add_artist(ell)
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell)
            ax.set_ylim(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return estimated, line, leftwheel, rightwheel, body, castor, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani


class Tricycle:
    """Tricycle or planar bicycle vehicle class.

    Parameters
    ----------
    ell_W : float
        The wheelbase of the vehicle [m].
    ell_T : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_W, ell_T):
        """Constructor method."""
        self.ell_W = ell_W
        self.ell_T = ell_T

    def f(self, x, u):
        """Tricycle or planar bicycle vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / self.ell_W * np.tan(x[3])
        f[3] = u[1]
        return f

    def draw(self, x, y, theta, phi):
        """Finds points that draw a tricycle vehicle.

        The centre of the rear wheel axle is (x, y), the body has orientation
        theta, steering angle phi, wheelbase ell_W and track length ell_T.

        Returns X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B, where L is for the left
        wheel, R is for the right wheel, F is for the single front wheel, and
        BD is for the vehicle's body.
        """
        # Left and right back wheels
        X_L, Y_L = graphics.draw_rectangle(
            x - 0.5 * self.ell_T * np.sin(theta),
            y + 0.5 * self.ell_T * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta,
        )
        X_R, Y_R = graphics.draw_rectangle(
            x + 0.5 * self.ell_T * np.sin(theta),
            y - 0.5 * self.ell_T * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta,
        )
        # Front wheel
        X_F, Y_F = graphics.draw_rectangle(
            x + self.ell_W * np.cos(theta),
            y + self.ell_W * np.sin(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta + phi,
        )
        # Body
        X_BD, Y_BD = graphics.draw_rectangle(
            x + self.ell_W / 2.0 * np.cos(theta),
            y + self.ell_W / 2.0 * np.sin(theta),
            2.0 * self.ell_W,
            2.0 * self.ell_T,
            theta,
        )
        # Return the arrays of points
        return X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD

    def animate(
        self,
        x,
        T,
        save_ani=False,
        filename="animate_tricycle.gif",
    ):
        """Create an animation of a tricycle vehicle.

        Returns animation object for array of vehicle positions x with time
        increments T [s], wheelbase ell_W [m], and track ell_T [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_tricycle.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (leftwheel,) = ax.fill([], [], color="k")
        (rightwheel,) = ax.fill([], [], color="k")
        (frontwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            leftwheel.set_xy(np.empty([5, 2]))
            rightwheel.set_xy(np.empty([5, 2]))
            frontwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, leftwheel, rightwheel, frontwheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the tricycle vehicle
            X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = self.draw(
                x[0, k], x[1, k], x[2, k], x[3, k]
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            frontwheel.set_xy(np.transpose([X_F, Y_F]))
            body.set_xy(np.transpose([X_B, Y_B]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_W, x[0, k] + 10 * self.ell_W)
            ax.set_ylim(x[1, k] - 10 * self.ell_W, x[1, k] + 10 * self.ell_W)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, leftwheel, rightwheel, frontwheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani


class Ackermann:
    """Ackermann steered vehicle class.

    Parameters
    ----------
    ell_W : float
        The wheelbase of the vehicle [m].
    ell_T : float
        The vehicle's track length [m].
    """

    def __init__(self, ell_W, ell_T):
        """Constructor method."""
        self.ell_W = ell_W
        self.ell_T = ell_T

    def f(self, x, u):
        """Ackermann steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's speed and steering angle rate.

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / self.ell_W * np.tan(x[3])
        f[3] = u[1]
        return f

    def ackermann(self, x):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_L = np.arctan(
            2 * self.ell_W *
            np.tan(x[3]) / (2 * self.ell_W - self.ell_T * np.tan(x[3]))
        )
        phi_R = np.arctan(
            2 * self.ell_W *
            np.tan(x[3]) / (2 * self.ell_W + self.ell_T * np.tan(x[3]))
        )
        ackermann_angles = np.array([phi_L, phi_R])
        return ackermann_angles

    def draw(self, x, y, theta, phi_L, phi_R):
        """Finds points that draw an Ackermann steered (car-like) vehicle.

        The centre of the rear wheel axle is (x, y), the body has orientation
        theta, effective steering angle phi, wheelbase ell_W and track length
        ell_T.

        Returns X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD,
        where L denotes left, R denotes right, B denotes back, F denotes front,
        and BD denotes the vehicle's body.
        """
        # Left and right back wheels
        X_BL, Y_BL = graphics.draw_rectangle(
            x - 0.5 * self.ell_T * np.sin(theta),
            y + 0.5 * self.ell_T * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta,
        )
        X_BR, Y_BR = graphics.draw_rectangle(
            x + 0.5 * self.ell_T * np.sin(theta),
            y - 0.5 * self.ell_T * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta,
        )
        # Left and right front wheels
        X_FL, Y_FL = graphics.draw_rectangle(
            x + self.ell_W * np.cos(theta) - self.ell_T / 2 * np.sin(theta),
            y + self.ell_W * np.sin(theta) + self.ell_T / 2 * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta + phi_L,
        )
        X_FR, Y_FR = graphics.draw_rectangle(
            x + self.ell_W * np.cos(theta) + self.ell_T / 2 * np.sin(theta),
            y + self.ell_W * np.sin(theta) - self.ell_T / 2 * np.cos(theta),
            0.5 * self.ell_T,
            0.25 * self.ell_T,
            theta + phi_R,
        )
        # Body
        X_BD, Y_BD = graphics.draw_rectangle(
            x + self.ell_W / 2.0 * np.cos(theta),
            y + self.ell_W / 2.0 * np.sin(theta),
            2.0 * self.ell_W,
            2.0 * self.ell_T,
            theta,
        )
        # Return the arrays of points
        return X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD

    def animate(
        self,
        x,
        T,
        phi_L,
        phi_R,
        save_ani=False,
        filename="animate_ackermann.gif",
    ):
        """Create an animation of an Ackermann steered (car-like) vehicle.

        Returns animation object for array of vehicle positions x with time
        increments T [s], wheelbase ell_W [m], and track ell_T [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default 'animate_ackermann.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (line,) = ax.plot([], [], "C0")
        (BLwheel,) = ax.fill([], [], color="k")
        (BRwheel,) = ax.fill([], [], color="k")
        (FLwheel,) = ax.fill([], [], color="k")
        (FRwheel,) = ax.fill([], [], color="k")
        (body,) = ax.fill([], [], color="C0", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """A function that initializes the animation."""
            line.set_data([], [])
            BLwheel.set_xy(np.empty([5, 2]))
            BRwheel.set_xy(np.empty([5, 2]))
            FLwheel.set_xy(np.empty([5, 2]))
            FRwheel.set_xy(np.empty([5, 2]))
            body.set_xy(np.empty([5, 2]))
            time_text.set_text("")
            return line, BLwheel, BRwheel, FLwheel, FRwheel, body, time_text

        def movie(k):
            """The function called at each step of the animation."""
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the Ackermann steered drive vehicle
            X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = self.draw(
                x[0, k], x[1, k], x[2, k], phi_L[k], phi_R[k]
            )
            BLwheel.set_xy(np.transpose([X_BL, Y_BL]))
            BRwheel.set_xy(np.transpose([X_BR, Y_BR]))
            FLwheel.set_xy(np.transpose([X_FL, Y_FL]))
            FRwheel.set_xy(np.transpose([X_FR, Y_FR]))
            body.set_xy(np.transpose([X_BD, Y_BD]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * self.ell_W, x[0, k] + 10 * self.ell_W)
            ax.set_ylim(x[1, k] - 10 * self.ell_W, x[1, k] + 10 * self.ell_W)
            ax.figure.canvas.draw()
            # Return the objects to animate
            return line, BLwheel, BRwheel, FLwheel, FRwheel, body, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000,
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani


class LongitudinalUSV:
    """3 DOF model of a USV (Surge, Heave, Pitch) for Longitudinal Control.

    Parameters
    ----------
    ell : float
        The hull length of the vehicle [m].
    """

    def __init__(self, ell):
        """Constructor method."""
        self.ell = ell

    def draw(self, x, y, theta):
        """
        Finds points that draw an uncrewed surface vessel.

        The centre of the vessel is (x, y), the vehicle has pitch angle
        theta, and the vehicle's hull length is ell.

        Returns X_ ...
        """
        L_TO_W = 4

        # Body
        X_B, Y_B = graphics.draw_rectangle(
            x, y, self.ell, (1/L_TO_W) * self.ell, theta,
        )

        # Mast/top of USV
        X_M, Y_M = graphics.draw_triangle(
            x - (1/(2*L_TO_W)) * self.ell * np.sin(theta),
            y + (1/(2*L_TO_W)) * self.ell * np.cos(theta),
            self.ell,
            0.5 * self.ell,
            theta - np.pi / 2,
        )
        # Return the arrays of points
        return X_B, Y_B, X_M, Y_M

    def animate(self, x, T, save_ani=False, filename="animate_longitudinalUSV.gif",
                relative=False):
        """Create an animation of an uncrewed surface vessel (longitudinal model).

        Returns animation object for array of vehicle positions x with time
        increments T [s], track ell [m].

        To save the animation to a GIF file, set save_ani to True and provide a
        filename (default 'animate_diffdrive.gif').
        """
        fig, ax = plt.subplots()
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.axis("equal")
        (body,) = ax.fill([], [], color="black")
        (mast,) = ax.fill([], [], color="orange")
        (wave,) = ax.plot([], [], color="silver")
        water = ax.fill_between([], [], [], color="deepskyblue", alpha=0.5)
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            """Function that initializes the animation."""
            # line.set_data([], [])
            body.set_xy(np.empty([5, 2]))
            mast.set_xy(np.empty([4, 2]))
            wave.set_data([], [])
            water.set_paths([])
            time_text.set_text("")
            return body, mast, wave, water, time_text

        def movie(k):
            """Function called at each step of the animation."""
            # Draw the path followed by the vehicle
            # line.set_data(x[0, 0: k + 1], x[1, 0: k + 1])
            # Draw the differential drive vehicle
            X_B, Y_B, X_M, Y_M = self.draw(
                x[0, k], x[1, k], x[2, k]
            )
            body.set_xy(np.transpose([X_B, Y_B]))
            mast.set_xy(np.transpose([X_M, Y_M]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.axis('equal')
            if relative:
                ax.set(xlim=(x[0, k] - 10 * self.ell, x[0, k] + 10 * self.ell),
                       ylim=(x[1, k] - 10 * self.ell, x[1, k] + 10 * self.ell))
            else:
                ax.set(xlim=(x[0, 0] - 10 * self.ell, x[0, 0] + 10 * self.ell),
                       ylim=(x[1, 0] - 10 * self.ell, x[1, 0] + 10 * self.ell))
            ax.figure.canvas.draw()
            # Return the objects to animate
            return body, mast, time_text

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            movie,
            np.arange(1, len(x[0, :]), max(1, int(1 / T / 10))),
            init_func=init,
            interval=T * 1000 * max(1, int(1 / T / 10)),
            blit=True,
            repeat=False,
        )
        if save_ani == True:
            ani.save(filename, fps=min(1 / T, 10))
        # Return the figure object
        return ani
