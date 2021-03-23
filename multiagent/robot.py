from multiagent.core import AgentState, Agent, World
import math
import numpy as np

class RobotState(AgentState):
    def __init__(self):
        super(RobotState, self).__init__()
        # state positions
        self.pos = []
        # length of robot arm
        self.lengths = []
        # resolution
        self.res = None
        # robot is grasping something
        self.grasp = False

        # # width of robot arm
        # self.width = None

class Robot(Agent):
    def __init__(self):
        super(Robot, self).__init__()
        # robot state
        self.state = RobotState()

    def create_robot_points(self):
        # returns a vector of the joint locations of a multiple joint robot arm
        points = [self.state.p_pos]
        lengths = self.state.lengths
        # cumulate state for defining relative joint positions
        cum_state = self.state.pos.cumsum()
        # resolution dependent step for rendering
        step = 2 * math.pi / self.state.res
        for i in range(len(cum_state)):
            joint_coordinates = [math.cos(step * cum_state[i]) * lengths[i],
                                 math.sin(step * cum_state[i]) * lengths[i]]
            points.append([sum(x) for x in zip(points[i], joint_coordinates)])
        return points

    def position_end_effector(self):
        # give the position of the end effector
        return self.create_robot_points()[-1]

    def within_reach(self, object):
        # test whether and object is within grasping range for a robot
        grasp_range = 0.05
        end_pos = np.array(self.position_end_effector())
        obj_pos = np.array(object.state.p_pos)
        dist = np.linalg.norm(obj_pos - end_pos)
        return dist <= grasp_range

class Roboworld(World):
    def __init__(self):
        super(Roboworld, self).__init__()
        # define arm length of robots
        self.arm_length = 10
        # joint per robot
        self.num_joints = 2

    def step(self):
        for agent in self.agents:
            self.update_agent_state(agent)
            self.update_object_state(agent, self.landmarks[0])

    def update_agent_state(self, agent):
        # print('State = ', agent.state.pos)
        # print('Action = ', agent.action.u)
        # print('length of pos vector', len(agent.state.pos))
        for i in range(len(agent.state.pos)): # 2 when agent has 2 joints
            agent.state.pos[i] += agent.action.u[i]
            # make sure state stays within resolution
            agent.state.pos[i] %= agent.state.res
            # print(agent.state.pos[i])

    def update_object_state(self, agent, object):
        if (agent.within_reach(object) == True) and (agent.state.grasp == True):
            object.state.p_pos = agent.position_end_effector()