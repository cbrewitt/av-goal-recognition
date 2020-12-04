# av-goal-recognition
Goal recognition for autonomous vehicles.

#Setup
Make sure you are using Python 3.6 or later.

Install Lanelet2 following the instructions [here](https://github.com/fzi-forschungszentrum-informatik/Lanelet2).

Clone this repository:
```
git clone https://github.com/cbrewitt/av-goal-recognition.git
```
Install with pip:
```
cd av-goal-recognition
pip install -e .
```

Extract the datasets into the same directory which `av-goal-recognition` is placed.
Alternatively, edit the config files in `av-goal-recognition/scenario_config` to specify different directories.

Preprocess the data and Extract features:

```
cd core
python data_processing.py
```

Train the decision trees:

```
cd ../decisiontree
python train_decision_tree.py
```

Calculate evaluation metrics on the test set:

```
cd ../evaluation/
python evaluate_models_from_features.py
```

Show animation of the dataset along with inferred goal probabilities:

```
python run_track_visualization.py --scenario heckstrasse --goal_recogniser trained_trees
```