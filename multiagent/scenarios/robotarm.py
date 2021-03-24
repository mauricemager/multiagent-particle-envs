import numpy as np
from multiagent.robot import Robot, Roboworld
from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        # define scenario properties
        num_agents = 1
        num_objects = 1
        # create world
        world = Roboworld()

        # add agents
        world.agents = [Robot() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

        # add objects
        world.objects = [Landmark() for i in range(num_objects)]
        for i, object in enumerate(world.objects):
            object.name = 'object %d' % i
            object.collide = True
            object.movable = True

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
        print(origins)
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.state.lengths = world.arm_length * np.ones(world.num_joints)
            agent.state.pos = np.random.randint(world.resolution, size=world.num_joints)
            agent.state.res = world.resolution
            agent.state.p_pos = np.array(origins[i][:])
            agent.state.p_vel = np.zeros(world.dim_p) # is needed
            # agent.state.c = np.zeros(world.dim_c) # not needed

        # set properties for landmarks
        for i, object in enumerate(world.objects):
            object.color = np.array([1, 0, 0])
            object.state.p_pos = 0.2 * np.random.randn(world.dim_p) + np.array([0.5, 0.0])
            object.state.p_vel = np.zeros(world.dim_p)

        # set properties for goal
        world.goals[0].state.p_pos = - world.objects[0].state.p_pos
        world.goals[0].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(world.objects[0].state.p_pos - world.goals[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.objects:
            dist = np.sum(np.square(entity.state.p_pos - agent.position_end_effector()))
            entity_pos = np.append(entity_pos, dist)
        obs = [[agent.state.pos], [entity_pos], [agent.state.grasp]]
        return obs


