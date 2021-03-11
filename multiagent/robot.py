
# ======================= Maurice =======================
from multiagent.core import AgentState, Agent, World

class RobotState(AgentState):
    def __init__(self):
        super(RobotState, self).__init__()
        # state positions
        self.pos = []
        # length of robot arm
        self.lengths = []
        # resolution
        self.res = None

        # # width of robot arm
        # self.width = None


class Robot(Agent):
    def __init__(self):
        super(Robot, self).__init__()
        # robot state
        self.state = RobotState()

class Roboworld(World):
    def __init__(self):
        super(Roboworld, self).__init__()
        # define arm length of robots
        self.arm_length = 10

    def step(self):
        for agent in self.agents:
            self.update_agent_state(agent)

    def update_agent_state(self, agent):
        print('State = ', agent.state.pos)
        print('Action = ', agent.action.u)
        # print('length of pos vector', len(agent.state.pos))
        for i in range(len(agent.state.pos)):
            agent.state.pos[i] += agent.action.u[i]
            # make sure state stays within resolution
            agent.state.pos[i] %= agent.state.res
            print(agent.state.pos[i])