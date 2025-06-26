from unray.envs.bridge_env import BridgeEnv
from unray.envs.bridge_env import MultiAgentBridgeEnv

class SingleAgentEnv():
    def __init__(self, config, env_name, isTuner = False, ip = 'localhost' , port = 9443, ID=1):
        self.env_ip = ip
        self.env_port = port
        self.env_config = config
        self.name = env_name
        self.ID = ID
        self.isTuner = isTuner

    def get_env(self):
        return lambda config: BridgeEnv(
            name = self.name, 
            ip = self.env_ip,
            port = self.env_port,
            config = self.env_config,
            first_connection = False,
            ID = self.ID,
            isTuner = self.isTuner,
        )
    
    def get_name(self):
        return self.name

    def isTuner(self):
        return self.isTuner
    
class MultiAgentEnv(SingleAgentEnv):
    def __init__(self, config, env_name, isTuner=False, ip='localhost', port=9443, ID = 1):
        super().__init__(config, env_name, isTuner, ip, port, ID)

    def get_env(self):
        return lambda config: MultiAgentBridgeEnv(
            name = self.name, 
            ip = self.env_ip,
            port = self.env_port,
            config = self.env_config,
            first_connection = False,
            ID = self.ID,
            isTuner = self.isTuner,
        )
    
    def get_name(self):
        return self.name
    
    def isTuner(self):
        return self.isTuner


