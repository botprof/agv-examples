# Autonomous Ground Vehicle Navigation & Control Simulation Examples in Python

__THIS PROJECT IS CURRENTLY A WORK IN PROGRESS AND THUS THIS REPOSITORY IS INCOMPLETE__

This is a repository of introductory autonomous ground vehicle (i.e., wheeled mobile robot) simulation examples in Python.  The purpose of these examples is to provide easy-to-follow code that is illustrative of a number of fundamental mobile robot modelling, control, and navigation (localization, mapping) concepts.  Motion planning problems lie beyond the scope of this example set.  The focus here is on ground vehicles, although the presented techniques are also applicable more broadly.  This code was initially developed to supplement topics covered in the course [ELEC 845 Autonomous Vehicle Control & Navigation](https://offroad.engineering.queensu.ca/courses/elec-845/) in the [Department of Electrical & Computer Engineering](http://www.ece.queensu.ca) at [Queen's University](http://www.queensu.ca).

## Requirements

The examples in this repository were created for use with [Python](https://www.python.org) 3.9.X or later.  The following packages are required in some or all of the examples in this repository:
* [NumPy](https://numpy.org) 1.22.X or later  
```pip install numpy``` or ```conda install numpy```
* [SciPy](https://scipy.org) 1.7.X or later  
```pip install scipy``` or ```conda install scipy```
* [Matplotlib](https://matplotlib.org) 3.5.X or later   
``` pip install matplotlib``` or ```conda install matplotlib```

The plotting routines also use [LaTeX](https://www.latex-project.org) for maths.  If you don't want to install LaTeX then you will have to comment out and edit those parts in the plotting routines.  However, using LaTeX is encouraged.

## MoBotPy Package

The repository also includes a supporting Python package MoBotPy (`mobotpy`) that contains some code that is used repeatedly.  Code developed in the worked examples is subsequently added to MoBotPy.

Module Filename | Description
--------------- | -----------
[integration.py](mobotpy/integration.py) | Provides basic Runge-Kutta and Euler integration functions.
[models.py](mobotpy/models.py) | Provides standard vehicle models, plotting, and animation methods.
[graphics.py](mobotpy/models.py) | Provides some basic shape plotting functions (used by [models.py](mobotpy/models.py)).

## Tables of Examples in this Repository

This section provides a list of the examples in this repository.

### Introductory Linear Systems Examples

These examples provide a review of basic concepts from control systems engineering in preparation for more advanced methods.

Filename | Description
-------- | -----------
[oneD_kinematic.py](oneD_kinematic.py) | Simulation of a linear 1D kinematic vehicle.
[oneD_dynamic.py](oneD_dynamic.py) | Simulation of a linear 1D dynamic vehicle.
[oneD_kinematic_control.py](oneD_kinematic_control.py) | Point stabilization of a linear 1D kinematic vehicle.
[oneD_dynamic_control.py](oneD_kinematic_control.py) | Point stabilization of a linear 1D dynamic vehicle.
[oneD_discrete_control.py](oneD_discrete_control.py) | Point stabilization of a linear 1D dynamic vehicle in discrete time.
[oneD_integral_control.py](oneD_integral_control.py) | Example illustrating integral action for disturbance rejection.
[oneD_dynamic_observer.py](oneD_dynamic_observer.py) | State estimation for a linear 1D dynamic vehicle.
[oneD_combined_control.py](oneD_combined_control.py) | Example illustrating control combined with a state estimator.

### Vehicle Modelling Examples

These examples provide simple models for a variety of commonly used wheeled vehicles.

Filename | Description
-------- | -----------
[diffdrive_kinematic.py](diffdrive_kinematic.py) | Simulation of a differential drive vehicle's kinematics.
[tricycle_kinematic.py](tricycle_kinematic.py) | Simulation of a tricycle vehicle's kinematics.
[ackermann_kinematic.py](ackermann_kinematic.py) | Simulation of an Ackermann steered (car-like) vehicle's kinematics.
[unicycle_dynamic.py](unicycle_dynamic.py) | Simulation of a dynamic unicycle (i.e, single wheel) illustrating wheel slip.

### Vehicle Control Examples

Filename | Description
-------- | -----------
[control_approx_linearization.py](control_approx_linearization.py) | Trajectory tracking for a differential drive vehicle using control by approximate linearization.
[dynamic_extension_tracking.py](dynamic_extension_tracking.py) | Trajectory tracking for a differential drive vehicle using feedback linearization with dynamic extension.
[MPC_linear_tracking.py](MPC_linear_tracking.py) | Trajectory tracking for a 1D dynamic vehicle using unconstrained model predictive control (MPC).

### Vehicle Navigation Examples

Filename | Description
-------- | -----------
[diffdrive_GNSS_EKF.py](diffdrive_GNSS_EKF.py) | Simple EKF implementation for a differential drive vehicle with wheel encoders, an angular rate gyro, and GNSS.
[UT_example.py](UT_example.py) | Introductory problem illustrating a basic unscented transform (UT) of statistics for Gaussian inputs, after [Julier and Uhlmann (2004)](https://doi.org/10.1109/JPROC.2003.823141).
[UKF_range_bearing.py](UKF_range_bearing.py) | Example implementation of a UKF for vehicle navigation by using odometry together with a range and bearing sensor, similar to the example on p. 290 of the book [Principles of Robot Motion: Theory, Algorithms, and Implementations (2005)](https://mitpress.mit.edu/books/principles-robot-motion).
[PF_range.py](PF_range.py) | Example implementation of a particle filter (PF) for vehicle navigation by using odometry together with a range sensor.  The example starts by showing particle clusters that grow with only dead reckoning, followed by a range-only sensor example with basic resampling.
[vanilla_SLAM.py](vanilla_SLAM.py) | Simple 2D SLAM example illustrating the basic notion of including feature measurements as part of the KF state.
 
## Cite this Work

You may wish to cite this work in your publications.  Use the appropriate [release version vX.X.X](https://github.com/botprof/agv-examples/releases) in your reference.

> Joshua A. Marshall, Autonomous Ground Vehicle Navigation and Control Simulation Examples in Python, vX.X.X, 2022, URL: [https://github.com/botprof/agv-examples](https://github.com/botprof/agv-examples).

You might also use the BibTeX entry below.

```latex
@misc{Marshall2021,
  author = {Marshall, Joshua A.},
  title = {Autonomous Ground Vehicle Navigation and Control Simulation Examples in Python, vX.X.X},
  year = {2022},
  howpublished = {\url{https://github.com/botprof/agv-examples}}
}
```
## Contact the Author

[Joshua A. Marshall](https://www.ece.queensu.ca/people/j-marshall), PhD, PEng  
[Department of Electrical & Computer Engineering](https://www.ece.queensu.ca)  
[Queen's University](http://www.queensu.ca)  
Kingston, ON K7L 3N6 Canada  
+1 (613) 533-2921  
[joshua.marshall@queensu.ca](mailto:joshua.marshall@queensu.ca)

## License

Source code examples in this notebook are subject to an [MIT License](LICENSE).
