"""
Python module models.py for various vehicle models.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

import numpy as np
from mobotpy import graphics
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Cart:
    """1D vehicle class (i.e., a simple cart).

    Parameters
    ----------
    length : float
        Length of the cart [m].
    """

    def __init__(self, length):
        """Constructor method."""
        self.length = length

    def draw(self, x, d):
        """Finds the points to draw simple rectangular cart.

        The cart has position x and length d.  The resulting cart has a height
        that is half the length.
        """
        X = np.array(5)
        Y = np.array(5)
        X = [x - d / 2, x - d / 2, x + d / 2, x + d / 2, x - d / 2]
        Y = [-d / 4, d / 4, d / 4, -d / 4, -d / 4]
        return X, Y

    def animate(self, x, T, d=1.0, save_ani=False, filename="animate_cart.gif"):
        """Create an animation of a simple 1D cart.

        Returns animation object for array of 1D cart positions x with time
        increments T [s], cart width d [m].

        To save the animation to a GIF file, set save_ani to True and give a
        filename (default filename is 'animate_cart.gif').
        """
        fig, ax = plt.subplots()
        plt.plot([np.min(x) - d, np.max(x) + d], [0, 0], "k--")
        plt.xlabel(r"$x$ [m]")
        ax.set_xlim([np.min(x) - d, np.max(x) + d])
        plt.yticks([])
        plt.axis("equal")
        (polygon,) = ax.fill([], [], "C0", alpha=0.5)
        (line,) = plt.plot([], [], "ko")
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        # Initialization funcion

        def init():
            polygon.set_xy(np.empty([5, 2]))
            line.set_data([], [])
            time_text.set_text("")
            return polygon, line, time_text

        # Function to draw cart

        def movie(k):
            X, Y = self.draw(x[k], d)
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

    def f(self, x, u, ell):
        """Differential drive kinematic vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 3
            The vehicle's state (x, y, theta).
        u : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        ell : float
            The vehicle's track length [m].

        Returns
        -------
        f : ndarray of length 3
            The rate of change of the vehicle states.
        """
        f = np.zeros(3)
        f[0] = 0.5 * (u[0] + u[1]) * np.cos(x[2])
        f[1] = 0.5 * (u[0] + u[1]) * np.sin(x[2])
        f[2] = 1.0 / ell * (u[1] - u[0])
        return f

    def uni2diff(self, u_in, ell):
        """
        Convert speed and anular rate inputs to differential drive wheel speeds.

        Parameters
        ----------
        u_in : ndarray of length 2
            The speed and turning rate of the vehicle (v, omega).
        ell : float
            The vehicle's track length [m].

        Returns
        -------
        u_out : ndarray of length 2
            The left and right wheel speeds (v_L, v_R).
        """
        v = u_in[0]
        omega = u_in[1]
        v_L = v - ell / 2 * omega
        v_R = v + ell / 2 * omega
        u_out = np.array([v_L, v_R])
        return u_out

    def draw(self, x, y, theta, ell):
        """
        Finds points that draw a differential drive vehicle.

        The centre of the wheel axle is (x, y), the vehicle has orientation
        theta, and the vehicle's track length is ell.

        Returns X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C, where L is for the
        left wheel, R for the right wheel, B for the body, and C for the caster.
        """
        # Left and right wheels
        X_L, Y_L = graphics.draw_rectangle(
            x - 0.5 * ell * np.sin(theta),
            y + 0.5 * ell * np.cos(theta),
            0.5 * ell,
            0.25 * ell,
            theta,
        )
        X_R, Y_R = graphics.draw_rectangle(
            x + 0.5 * ell * np.sin(theta),
            y - 0.5 * ell * np.cos(theta),
            0.5 * ell,
            0.25 * ell,
            theta,
        )
        # Body
        X_BD, Y_BD = graphics.draw_circle(x, y, ell)
        # Caster
        X_C, Y_C = graphics.draw_circle(
            x + 0.5 * ell * np.cos(theta), y + 0.5 * ell * np.sin(theta), 0.125 * ell
        )
        # Return the arrays of points
        return X_L, Y_L, X_R, Y_R, X_BD, Y_BD, X_C, Y_C

    def animate(self, x, T, ell=1.0, save_ani=False, filename="animate_diffdrive.gif"):
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
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k], ell
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * ell, x[0, k] + 10 * ell)
            ax.set_ylim(x[1, k] - 10 * ell, x[1, k] + 10 * ell)
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
        self, x, xd, T, ell=1.0, save_ani=False, filename="animate_diffdrive.gif"
    ):
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
            desired.set_data(xd[0, 0 : k + 1], xd[1, 0 : k + 1])
            # Draw the path followed by the vehicle
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the differential drive vehicle
            X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = self.draw(
                x[0, k], x[1, k], x[2, k], ell
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            body.set_xy(np.transpose([X_B, Y_B]))
            castor.set_xy(np.transpose([X_C, Y_C]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * ell, x[0, k] + 10 * ell)
            ax.set_ylim(x[1, k] - 10 * ell, x[1, k] + 10 * ell)
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

    def f(self, x, u, ell_W):
        """Tricycle or planar bicycle vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's speed and steering angle rate.
        ell_W : float
            The vehicle's wheelbase [m].

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / ell_W * np.tan(x[3])
        f[3] = u[1]
        return f

    def draw(self, x, y, theta, phi, ell_W, ell_T):
        """Finds points that draw a tricycle vehicle.

        The centre of the rear wheel axle is (x, y), the body has orientation
        theta, steering angle phi, wheelbase ell_W and track length ell_T.

        Returns X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B, where L is for the left
        wheel, R is for the right wheel, F is for the single front wheel, and
        BD is for the vehicle's body.
        """
        # Left and right back wheels
        X_L, Y_L = graphics.draw_rectangle(
            x - 0.5 * ell_T * np.sin(theta),
            y + 0.5 * ell_T * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta,
        )
        X_R, Y_R = graphics.draw_rectangle(
            x + 0.5 * ell_T * np.sin(theta),
            y - 0.5 * ell_T * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta,
        )
        # Front wheel
        X_F, Y_F = graphics.draw_rectangle(
            x + ell_W * np.cos(theta),
            y + ell_W * np.sin(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta + phi,
        )
        # Body
        X_BD, Y_BD = graphics.draw_rectangle(
            x + ell_W / 2.0 * np.cos(theta),
            y + ell_W / 2.0 * np.sin(theta),
            2.0 * ell_W,
            2.0 * ell_T,
            theta,
        )
        # Return the arrays of points
        return X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD

    def animate(
        self,
        x,
        T,
        ell_W=1.0,
        ell_T=1.0,
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
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the tricycle vehicle
            X_L, Y_L, X_R, Y_R, X_F, Y_F, X_B, Y_B = self.draw(
                x[0, k], x[1, k], x[2, k], x[3, k], ell_W, ell_T
            )
            leftwheel.set_xy(np.transpose([X_L, Y_L]))
            rightwheel.set_xy(np.transpose([X_R, Y_R]))
            frontwheel.set_xy(np.transpose([X_F, Y_F]))
            body.set_xy(np.transpose([X_B, Y_B]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * ell_W, x[0, k] + 10 * ell_W)
            ax.set_ylim(x[1, k] - 10 * ell_W, x[1, k] + 10 * ell_W)
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

    def f(self, x, u, ell_W):
        """Ackermann steered vehicle kinematic model.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        u : ndarray of length 2
            The vehicle's speed and steering angle rate.
        ell_W : float
            The vehicle's wheelbase [m].

        Returns
        -------
        f : ndarray of length 4
            The rate of change of the vehicle states.
        """
        f = np.zeros(4)
        f[0] = u[0] * np.cos(x[2])
        f[1] = u[0] * np.sin(x[2])
        f[2] = u[0] * 1.0 / ell_W * np.tan(x[3])
        f[3] = u[1]
        return f

    def ackermann(self, x, ell_W, ell_T):
        """Computes the Ackermann steering angles.

        Parameters
        ----------
        x : ndarray of length 4
            The vehicle's state (x, y, theta, phi).
        ell_W : float
            The vehicle's wheelbase [m].
        ell_T :float
            The vehicle's track length [m].

        Returns
        -------
        ackermann_angles : ndarray of length 2
            The left and right wheel angles (phi_L, phi_R).
        """
        phi_L = np.arctan(2 * ell_W * np.tan(x[3]) / (2 * ell_W - ell_T * np.tan(x[3])))
        phi_R = np.arctan(2 * ell_W * np.tan(x[3]) / (2 * ell_W + ell_T * np.tan(x[3])))
        ackermann_angles = np.array([phi_L, phi_R])
        return ackermann_angles

    def draw(self, x, y, theta, phi_L, phi_R, ell_W, ell_T):
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
            x - 0.5 * ell_T * np.sin(theta),
            y + 0.5 * ell_T * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta,
        )
        X_BR, Y_BR = graphics.draw_rectangle(
            x + 0.5 * ell_T * np.sin(theta),
            y - 0.5 * ell_T * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta,
        )
        # Left and right front wheels
        X_FL, Y_FL = graphics.draw_rectangle(
            x + ell_W * np.cos(theta) - ell_T / 2 * np.sin(theta),
            y + ell_W * np.sin(theta) + ell_T / 2 * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta + phi_L,
        )
        X_FR, Y_FR = graphics.draw_rectangle(
            x + ell_W * np.cos(theta) + ell_T / 2 * np.sin(theta),
            y + ell_W * np.sin(theta) - ell_T / 2 * np.cos(theta),
            0.5 * ell_T,
            0.25 * ell_T,
            theta + phi_R,
        )
        # Body
        X_BD, Y_BD = graphics.draw_rectangle(
            x + ell_W / 2.0 * np.cos(theta),
            y + ell_W / 2.0 * np.sin(theta),
            2.0 * ell_W,
            2.0 * ell_T,
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
        ell_W=1.0,
        ell_T=1.0,
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
            line.set_data(x[0, 0 : k + 1], x[1, 0 : k + 1])
            # Draw the Ackermann steered drive vehicle
            X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = self.draw(
                x[0, k], x[1, k], x[2, k], phi_L[k], phi_R[k], ell_W, ell_T
            )
            BLwheel.set_xy(np.transpose([X_BL, Y_BL]))
            BRwheel.set_xy(np.transpose([X_BR, Y_BR]))
            FLwheel.set_xy(np.transpose([X_FL, Y_FL]))
            FRwheel.set_xy(np.transpose([X_FR, Y_FR]))
            body.set_xy(np.transpose([X_BD, Y_BD]))
            # Add the simulation time
            time_text.set_text(r"$t$ = %.1f s" % (k * T))
            # Dynamically set the axis limits
            ax.set_xlim(x[0, k] - 10 * ell_W, x[0, k] + 10 * ell_W)
            ax.set_ylim(x[1, k] - 10 * ell_W, x[1, k] + 10 * ell_W)
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
