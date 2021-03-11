import numpy as np
import math
from multiagent.robot import Robot, Roboworld
from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = Roboworld()
        num_agents = 1
        num_landmarks = 1
        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # properties for multiple agents evenly spaced on a circle
        radius_agent = 0.5
        angle = 2*math.pi / len(world.agents)
        num_joints = 1
        resolution = 8
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = 0.25*np.ones(num_joints)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([1,0,0])
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.pos = np.random.randint(resolution, size=num_joints)
            print(agent.state.pos)
            # agent.state.angles = np.random.uniform(0,2 * math.pi, num_joints)
            # agent.state.angles = np.zeros(num_joints)
            agent.state.res = resolution
            agent.state.p_pos = np.array([radius_agent * math.cos(angle * i),
                                          radius_agent * math.sin(angle * i)])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([0,0])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
