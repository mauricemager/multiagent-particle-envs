#!/usr/bin/env python
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.robot_environment import MultiAgentEnv
from multiagent.robot_policy import RobotInteractivePolicy
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

"""
################################# TASKS #################################
* make rendering possible for 2 arms 
"""
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='robotarm.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [RobotInteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        print("act_n", act_n)
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        print('End effector position test: ', world.agents[1].position_end_effector())

        print('You can grab to object: ', world.agents[1].test_object_graspable(world.landmarks[0]))

        print('Agent is grabbing:', world.agents[1].state.grasp)

        print('Position')