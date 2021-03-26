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


class Robot(Agent):
    def __init__(self):
        super(Robot, self).__init__()
        # robot state
        self.state = RobotState()

    def create_robot_points(self, shorter_end=False):
        # returns a vector of the joint locations of a multiple joint robot arm
        points = [self.state.p_pos]
        lengths = self.state.lengths
        # cumulate state for defining relative joint positions
        cum_state = self.state.pos.cumsum()
        # resolution dependent step for rendering
        step = 2 * math.pi / self.state.res
        for i in range(len(cum_state)):
            length = lengths[i]
            # remove part of last arm for better rendering with gripper
            if shorter_end and (i == range(len(cum_state))[-1]):
                length -= 0.045
            # joint coordinates per segment
            joint_coordinates = [math.cos(step * cum_state[i]) * length,
                                 math.sin(step * cum_state[i]) * length]
            # add the joint coordinates to the points vector
            points.append([sum(x) for x in zip(points[i], joint_coordinates)])
        return points

    def create_gripper_points(self, radius=0.05, res=30, gripped=False):
        # return a vector of the gripper points for rendering
        # angle of gripper clearance
        phi = math.pi / 3
        points = list()
        # orientation of the gripper w.r.t. end effector
        orientation = 2 * math.pi * np.sum(self.state.pos) / self.state.res
        # smaller gripper and clearance when gripped
        if gripped:
            radius *= 0.8
            phi /= 3
        # create the gripper points relative to end effector orientation
        for i in range(res):
            ang = 2 * math.pi * i / res
            if (ang >= 0.5 * phi) and (ang <= 2 * math.pi - 0.5 * phi):
                points.append([math.cos(ang + orientation) * radius,
                               math.sin(ang + orientation) * radius])
        # translate gripped points to end effector location
        points = np.array(points) + [self.position_end_effector()]
        return points

    def position_end_effector(self):
        # give the position of the end effector
        return np.array(self.create_robot_points()[-1])

    def within_reach(self, object, grasp_range=0.05):
        # test whether and object is within grasping range for a robot
        end_pos = np.array(self.position_end_effector())
        obj_pos = np.array(object.state.p_pos)
        dist = np.linalg.norm(obj_pos - end_pos)
        return dist <= grasp_range


class Roboworld(World):
    def __init__(self):
        super(Roboworld, self).__init__()
        # define arm length of robots
        self.arm_length = 0.35
        # joint per robot
        self.num_joints = 2
        #
        self.goals = []
        #
        self.resolution = 180

    @property
    def entities(self):
        return self.agents + self.objects + self.goals

    def step(self):
        for agent in self.agents:
            self.update_agent_state(agent)
            self.update_object_state(agent, self.objects[0])

    def update_agent_state(self, agent):
        # print('State = ', agent.state.pos)
        # print('Action = ', agent.action.u)
        # print('length of pos vector', len(agent.state.pos))
        for i in range(len(agent.state.pos)):  # 2 when agent has 2 joints
            agent.state.pos[i] += agent.action.u[i]
            # make sure state stays within resolution
            agent.state.pos[i] %= agent.state.res
            # print(agent.state.pos[i])

    def update_object_state(self, agent, object):
        if (agent.within_reach(object) == True) and (agent.state.grasp == True):
            object.state.p_pos = agent.position_end_effector()

    def robot_position(self, n, r=0.5):
        if n == 1:
            return [[0, 0]]
        else:
            phi = 2 * math.pi / n
            position = [[r * math.cos(phi * i + math.pi),
                         r * math.sin(phi * i + math.pi)] for i in range(n)]
            return position
