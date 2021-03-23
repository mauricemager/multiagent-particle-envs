import numpy as np
import math
from multiagent.robot import Robot, Roboworld
from multiagent.core import Landmark, Entity
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 2
        num_landmarks = 1
        # create world
        world = Roboworld()

        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'object %d' % i
            landmark.collide = True
            landmark.movable = True

        # add goals
        world.goals = [Landmark() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal'
            goal.collide = False
            goal.movable = False

        # reset world
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set agent properties
        origins = world.robot_position(len(world.agents))
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.arm_length * np.ones(world.num_joints)
            agent.state.pos = np.random.randint(world.resolution, size=world.num_joints)
            agent.state.res = world.resolution
            agent.state.p_pos = origins[i][:]
            agent.state.p_vel = np.zeros(world.dim_p) # is needed
            # agent.state.c = np.zeros(world.dim_c) # not needed

        # set properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1, 0, 0])
            landmark.state.p_pos = 0.3 * np.random.randn(world.dim_p) - np.array([0.5, 0.0])
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set properties for goal
        world.goals[0].state.p_pos = - world.landmarks[0].state.p_pos
        world.goals[0].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(world.landmarks[0].state.p_pos - world.goals[0].state.p_pos))
        # dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)


