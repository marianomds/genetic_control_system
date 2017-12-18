# genetic_control_system
Control system parameters optimization with a genetic algorithm. Results are sored in a database using SQLObject.

Dependencies
============

- SQLObject
- python-control: https://github.com/python-control/python-control
- Matplotlib
- Numpy
- Scipy
- MySQL (or other database engine compatible with SQLObject)

Instructions
============

- Run GeneticPID.sql script for creating database.
- Each time main.py is run, it's configuration and results are stored in the database.
