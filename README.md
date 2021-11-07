# Autonomous Ground Vehicle Navigation and Control Simulation Examples in Python

__THIS PROJECT IS CURRENTLY A WORK IN PROGRESS AND THUS THIS REPOSITORY IS INCOMPLETE__

This is a repository of introductory autonomous ground vehicle (i.e., wheeled mobile robot) simulation examples in [Python](https://www.python.org).  The purpose of these examples is to provide easy-to-follow code that is illustrative of a number of fundamental mobile robot modelling, control, and navigation (localization, mapping) concepts.  Motion planning problems lie beyond the scope of this example set.  The focus here is on ground vehicles, although the presented techniques are also applicable more broadly.  This code was initially developed to supplement topics covered in the course [ELEC 845 Autonomous Vehicle Control & Navigation](https://offroad.engineering.queensu.ca/courses/elec-845/) in the [Department of Electrical & Computer Engineering](http://www.ece.queensu.ca) at [Queen's University](http://www.queensu.ca).

## MoBotPy Package

The repository also includes a supporting Python package MoBotPy (`mobotpy`) that contains some code that is used repeatedly.  Code developed in the worked examples is subsequently added to MoBotPy.

Module Filename | Description
--------------- | -----------
[integration.py](mobotpy/integration.py) | Provides basic Runge-Kutta and Euler integration functions.
[models.py](mobotpy/models.py) | Provides standard vehicle models, plotting, and animation methods.
[graphics.py](mobotpy/models.py) | Provides some basic shape plotting functions (used by [models.py](mobotpy/models.py)).

## Tables of Examples in this Repository

This section provides a list of the examples in this repository.

### Introductory Linear System Examples

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
 
## Cite this Work

You may wish to cite this work in your publications.

> Joshua A. Marshall, Autonomous Ground Vehicle Navigation and Control Simulation Examples in Python, 2021, URL: [https://github.com/botprof/agv-examples](https://github.com/botprof/agv-examples).

You might also use the BibTeX entry below.

```latex
@misc{Marshall2021,
  author = {Marshall, Joshua A.},
  title = {Autonomous Ground Vehicle Navigation and Control Simulation Examples in Python},
  year = {2021},
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
