# Robot Learning: Autonomous Driving with Imitation Learning

This repository showcases a project focused on implementing and exploring various imitation learning models for autonomous driving, including intent inference and shared autonomy. The project utilizes a partially observable Markov decision process (POMDP) framework and the CARLO (CARLA - Low Budget) 2D driving simulator for experiments.

## Project Goals

The primary objectives of this project were to:

* **Implement Behavior Cloning:** Develop a foundational behavior cloning model to learn deterministic driving policies directly from expert demonstrations.
* **Explore Distributional Policies with Mixture Density Networks:** Extend the behavior cloning approach to learn a distribution over actions using Mixture Density Networks. This is crucial for more sophisticated applications like inferring user intent.
* **Develop Conditional Imitation Learning (CoIL):** Implement a CoIL strategy to create a single, robust policy capable of handling multiple high-level driving goals (e.g., turning left, right, or going straight at an intersection).
* **Apply Intent Inference & Shared Autonomy:** Utilize the learned distributional policies to predict user intent in real-time and enable a shared control system between a human driver and the autonomous agent.

## Driving Scenarios

The project utilizes two distinct driving scenarios within the CARLO simulator:

* **Intersection Scenario:** The agent's task is to navigate an intersection, choosing to turn left, go straight, or turn right, while avoiding collisions with pedestrians and buildings. The intersection's placement is dynamic, changing with each simulation run. The agent receives observations including its car's position, speed, heading angle, and the intersection's location.
* **Lanechange Scenario:** The objective is to follow any lane on a straight road without departing from the road. Observations provided to the agent include the car's position, speed, and heading angle.

## Technical Setup

The project was primarily developed using Google Colab. All necessary Python dependencies are installed via the provided code.

To set up the environment and run the code:

1.  Clone this repository:
    ```bash
    git clone https://github.com/USC-Lira/CSC1699_Robot_Learning_HW3.git # (Replace with your actual repo URL)
    ```
2.  Navigate to the cloned directory. The project's Python scripts are designed to install required packages automatically upon execution.

## Running the Demonstrations

The repository includes scripts to interact with the CARLO simulator and demonstrate the trained models.

### Interactive Simulation

You can interactively observe the CARLO simulator in action:
```bash
python play.py --scenario intersection
# or
python play.py --scenario lanechange
```

### Behavior Cloning (Deterministic Policies)

**Training a Deterministic Policy (`train_il.py`):**
```bash
python train_il.py --scenario intersection --goal left --epochs <number_of_epochs> --lr <learning_rate>
# Replace 'left' with 'straight' or 'right' to train for other goals.
# Use --restore to continue training from a previously saved policy checkpoint.
```

**Testing a Deterministic Policy (`test_il.py`):**
```bash
python test_il.py --scenario intersection --goal left --visualize
# To obtain a numerical success rate without visualization:
python test_il.py --scenario intersection --goal left
```

### Behavior Cloning (Distributional Policies with MDN)

**Training a Distributional Policy (`train_ildist.py`):**
```bash
python train_ildist.py --scenario intersection --goal left --epochs <number_of_epochs> --lr <learning_rate>
# Replace 'left' with 'straight' or 'right' for other goals.
# Use --restore to resume training from a checkpoint.
```

**Testing a Distributional Policy (`test_ildist.py`):
```bash
python test_ildist.py --scenario intersection --goal left --visualize
# To obtain a numerical success rate without visualization:
python test_ildist.py --scenario intersection --goal left
```

### Conditional Imitation Learning (CoIL)

**Training a CoIL Policy (`train_coil.py`):**
```bash
python train_coil.py --scenario intersection --epochs <number_of_epochs> --lr <learning_rate>
# Use --restore to resume training.
```

**Testing a CoIL Policy (`test_coil.py`):**
```bash
python test_coil.py --scenario intersection --visualize
# During visualization, control the high-level commands (goals) using arrow keys.
# To obtain a numerical success rate for a specific goal:
python test_coil.py --scenario intersection --goal left
# Repeat for 'straight' and 'right' goals to evaluate performance across all commands.
```

### Intent Inference & Shared Autonomy

**Intent Inference Demonstration (`intent_inference.py`):**
```bash
python intent_inference.py --scenario intersection
# Drive the car using arrow keys to observe real-time intent predictions from the model.
```

**Shared Autonomy Demonstration (`shared_autonomy.py`):**
First, ensure that the Mixture Density Networks (`train_ildist.py`) for the `lanechange` scenario are trained for both `left` and `right` goals:
```bash
python train_ildist.py --scenario lanechange --goal left --epochs <number_of_epochs> --lr <learning_rate>
python train_ildist.py --scenario lanechange --goal right --epochs <number_of_epochs> --lr <learning_rate>
```
Then, execute the shared autonomy script:
```bash
python shared_autonomy.py --scenario lanechange
# In this mode, you control the vehicle's steering with arrow keys, while the throttle is automatically managed by the shared autonomy system.
```

## Learned Policies

All trained policy models are saved in the `policies/` directory within this repository.

## References

* [1] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. [cite_start]In Conference on Robot Learning, 2017. [cite: 1]
* [2] Zhangjie Cao, Erdem Bıyık, Woodrow Z Wang, Allan Raventos, Adrien Gaidon, Guy Rosman, and Dorsa Sadigh. Reinforcement learning based control of imitative policies for near-accident driving. [cite_start]Proceedings of Robotics: Science and Systems (RSS), 2020. [cite: 2]
* [3] Jason Kong, Mark Pfeiffer, Georg Schildbach, and Francesco Borrelli. Kinematic and dynamic vehicle models for autonomous driving control design. [cite_start]In IEEE Intelligent Vehicles Symposium (IV), 2015. [cite: 3]
* [4] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. [cite_start]Advances in Neural Information Processing Systems, 1988. [cite: 4]
* [5] Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. [cite_start]In IEEE International Conference on Robotics and Automation (ICRA), 2018. [cite: 5]
