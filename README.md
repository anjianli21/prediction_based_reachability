Prediction-Based Reachability for Collision Avoidance in Autonomous Driving
==========

This is the codebase for the paper ["Predicion-Based Reachability for Collision Avoidance in Autonomous Driving"](https://arxiv.org/abs/2011.12406). In this codebase, we provide the code to cluster trajectory data into driving modes, to compute reachable tubes for each mode, and to simulate the two-car collision avoidance using safety controller.

This codebase is built on [optimized_dp](https://github.com/SFU-MARS/optimized_dp) toolbox, [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) codebase, and uses [Interaction](http://interaction-dataset.com/) dataset.

## Setup

Please first read the setup procedure for [optimized_dp](https://github.com/SFU-MARS/optimized_dp) toolbox.

Besides, please install the below packages using conda.

```
pickle
numpy
matplotlib
scikit-learn
pandas
pillow
shutil
json
```
****
## Content

This codebase contains three parts: 

1. Process the predicted trajectory data and cluster it into different driving mode, which includes:
```
~/prediction/

```

2. Simulate the two-car collision avoidance
```
~/simulation_stanley/
~/simulation_helper/
```

3. Compute reachable tubes for each driving mode
```
rest of the code
```

## Trajectory processing, driving mode clustering and mode prediction

First, configure the data path in "def __init__(self): " in both process_prediction.py and predict_mode.py.

1. Process the predicted trajectory data and obtain actions:
```
python prediction/process_prediction.py

```

2. Cluster the action into driving mode
```
python prediction/clustering.py

```

3. Predict driving mode giving a new trajectory
```
python prediction/predict_mode.py

```

## Reachable tube computation

1. Compute reachable tube for 5d relative dynamics
```
python solver_reldyn5d.py

```

2. Compute reachable tube for 4d bicycle dynamics (for curbs)
```
python solver_bicycle4d.py

```

## Simulation

Configure the data path in "def __init__(self):" in simulator_stanley_helper.py.

Configure the scenario and trial that you want to simulate in "def reset_trial(self, trial_name, scenario):" in simulator_stanley.py.

Run the simulation:
```
python simulator_stanley.py

```

## Citing This Work
If you use this work in your research please cite:
```
@article{li2020prediction,
  title={Prediction-Based Reachability for Collision Avoidance in Autonomous Driving},
  author={Li, Anjian and Sun, Liting and Zhan, Wei and Tomizuka, Masayoshi and Chen, Mo},
  journal={arXiv preprint arXiv:2011.12406},
  year={2020}
}
```
