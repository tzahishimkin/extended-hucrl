# Implementation of H-UCRL Algorithm

[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/hucrl/master?label=master%20build%20and%20test&token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://app.circleci.com/pipelines/github/sebascuri/hucrl)
[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/hucrl/dev?label=dev%20build%20and%20test&token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://app.circleci.com/pipelines/github/sebascuri/hucrl)
[![CircleCI](https://circleci.com/gh/sebascuri/hucrl/tree/master.svg?style=shield&circle-token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://circleci.com/gh/circleci/circleci-docs/tree/teesloane-patch-5)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/hug/)

This repo is an implementation of the H-UCRL algorithm introduced in
Curi, S., Berkenkamp, F., & Krause, A. (2020). Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning.



To install create a conda environment:
```bash
$ conda create -n hucrl python=3.7
$ conda activate hucrl
```

```bash
$ pip install -e .[test,logging,experiments]
```

For Mujoco (license required) Run:
```bash
$ pip install -e .[mujoco]
```

On clusters run:
```bash
$ sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth
```


## Running an experiment.
```bash
$ python exps/ENVIRONMENT/mbmpo.py
```

For example, for the inverted pendulum experiment run
```bash
$ python exps/inverted_pendulum/mbmpo.py
```

## Pre Commit
install pre-commit with
```bash
$ pip install pre-commit
$ pre-commit install
```

Run pre-commit with
```bash
$ pre-commit run --all-files
```
