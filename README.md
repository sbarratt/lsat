# lsat

Code accompanying the paper [Least Squares Auto-Tuning](http://web.stanford.edu/~boyd/papers/lsat.html).

## Installation

We use [conda](https://conda.io/miniconda.html) for python package management.
To install our `lsat` environment, run the following commands while in the repository:
```
$ conda env create -f environment.yml
$ conda activate lsat
```

## Tests
To run tests, run (with the `lsat` environment activated):
```
$ python lstsq.py
```
You should see "All tests passed!".

## MNIST example
To run the MNIST example described in the paper,
type into a terminal (with the `lsat` environment activated):
```
$ python mnist.py # to download the data
$ python mnist_example.py # to run the script
```
