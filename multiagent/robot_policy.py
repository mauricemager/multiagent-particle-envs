from multiagent.policy import *

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class RobotInteractivePolicy(InteractivePolicy):
    def __init__(self):
        super(RobotInteractivePolicy, self).__init__(env, agent_index)