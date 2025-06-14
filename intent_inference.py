#!/usr/bin/env python3
import numpy as np
import gym_carlo
import gym
import time
import argparse
import torch
from gym_carlo.envs.interactive_controllers import KeyboardController
from scipy.stats import multivariate_normal
from train_ildist import MDN
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="intersection, circularroad, lanechange",
        default="intersection",
    )
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, "--scenario argument is invalid!"

    env = gym.make(scenario_name + "Scenario-v0", goal=len(goals[scenario_name]))

    mdns = {}
    for goal in goals[scenario_name]:
        mdns[goal] = MDN(obs_sizes[scenario_name], 2)
        ckpt_path = "./policies/" + scenario_name + "_" + goal + "_ILDIST"
        mdns[goal].load_state_dict(torch.load(ckpt_path))

    env.T = 200 * env.dt - env.dt / 2.0  # Run for at most 200dt = 20 seconds
    for _ in range(15):
        env.seed(int(np.random.rand() * 1e6))
        obs, done = env.reset(), False
        env.render()
        interactive_policy = KeyboardController(env.world, steering_lims[scenario_name])
        scores = np.array([0.0] * len(goals[scenario_name]))
        while not done:
            t = time.time()

            # Get the current observation and action
            if scenario_name == "intersection":
                action = [interactive_policy.steering, interactive_policy.throttle]
            elif scenario_name == "circularroad":
                opt_action = optimal_act_circularroad(
                    env, 0
                )  # desired lane doesn't matter for throttle
                action = [interactive_policy.steering, opt_action[0, 1]]
            elif scenario_name == "lanechange":
                opt_action = optimal_act_lanechange(
                    env, 0
                )  # desired lane doesn't matter for throttle
                action = [interactive_policy.steering, opt_action[0, 1]]
            obs = np.array(obs).reshape(1, -1)

            # We compute the probabilities of each goal based on the (observation,action) pair, i.e. we will compute P(g | o, a) for all g.

            # - goals[scenario_name] is the list of goals, e.g. ['left','straight','right'] for the intersection scenario
            # - nn_models[goal] is the trained mixed density network, e.g. nn_models['left']
            # - obs (1 x dim(O) numpy array) is the current observation
            # - action (1 x 2 numpy array) is the current action the user took when the observation is obs

            probs = []
            for goal in goals[scenario_name]:
                out = mdns[goal].forward(torch.Tensor(obs)).detach().numpy()
                #print(out)
                out = np.reshape(out, (6,))
                probs.append(multivariate_normal(out[:2], np.reshape(out[2:], (2, 2))).logpdf(action))
            probs = np.array(probs)
        

            # Print the prediction on the simulator window. This is also scenario-dependent.
            if (
                scenario_name == "intersection" and env.ego_approaching_intersection
            ) or scenario_name in ["circularroad", "lanechange"]:
                env.write(
                    "Inferred Intent: Going "
                    + goals[scenario_name][np.argmax(probs)]
                    + "\nConfidence = {:.2f}".format(np.max(probs) / np.sum(probs))
                )
                scores += probs
            elif (
                np.sum(scores) > 1
            ):  # the car has passed the intersection for the IntersectionScenario-v0
                env.write(
                    "Overall prediction: Going "
                    + goals[scenario_name][np.argmax(scores)]
                    + "\nConfidence = {:.2f}".format(np.max(scores) / np.sum(scores))
                )
            obs, _, done, _ = env.step(action)
            env.render()
            while time.time() - t < env.dt / 2:
                pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                time.sleep(1)
