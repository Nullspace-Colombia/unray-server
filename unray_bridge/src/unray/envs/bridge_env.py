""" 
    BridgeEnv

    Definition of Main Bridge Envs that allow to communicate 
    with socket to send actions and receive observations from 
    UnrealEngine5 (UE5)

    There are two (2) main BridgeEnvironments. The SingleAgent 
    and the MultiAgent Environment 
    
    -   BridgeEnv 
    -   MultiAgentBridgeEnv

    You have to define a observation_space and actions_space from 
    BridgeSpaces (@see unray_bridge.envs.spaces).

    @author Valentina Hernandez
    @author Andrés Morales 

    @version: 0.1V
    
    @file bridge_env.py 

"""
from .bridge.TCP_IP_Connector import ClientHandler
from unray.envs.spaces import BridgeSpaces
from gymnasium import Env as gymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from socket import socket
from unray.envs.bridge.TCP_IP_Connector import ClientHandler

class BridgeEnv(gymEnv): 
    """
        Base class for custom UE5 Conenction. 
    """
    # 1. Custom gymenv metadata for viz. 
    metadata = {
        "rendermodes": [""], 
        "render_fps": ""
    }

    def __init__(self, 
                 name, 
                 ip, 
                 port, 
                 config, 
                 first_connection = False, 
                 validation = False, 
                 multiagent = False,
                 ID = int,
                 isTuner = False):
                 
        
        self.ip = ip # IP Address for IP Connection 
        self.port = port 
        self.ID = ID

        self.client_handler = ClientHandler(ip, port + ID)

        if not name:
            print("error")
            raise ValueError("no environment name defined. Please define a name for tour environment on the BridgeEnv constructor.")
        self.name = name # environment name for later id 
        self.has_handler = False # ClientHandler to begin with connection
        self.has_connection = False
        self.dummy_action = None
        self.validation = validation
        self.multiagent = multiagent 
        self.reset_count = -2
        
        try: 
            self.observation_space = config["observation"] # Get imported space 
            self.action_space = config["action"]
        except: 
            raise ValueError("No correct space selected")

        self.obs = self.observation_space.sample()

        self.create_handler()

        if isTuner:
            self.connect_socket()

    
    def step(self, action): 

        #if not self.has_connection:
        #    self.connect()

        action=np.array(action,dtype=np.single)

        if isinstance(action, list): 
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            pass
        else:
            assert "No valid action type. Only supports <list> or <numpy.ndarray> given %s" % (type(action))
        

        state = np.empty(self.observation_space.shape, dtype=self.observation_space.dtype)
        action_buff = action.tobytes()
        n = self.get_amount_obs()

       
        if self.reset_count > 0:
            
        # 2. Cast the action vector to a byte buffer for send.
            
        # 3. Send the action vector to the environment. 
        #self.handler.send(action_buff) # Send action an wait response 
            data_size = 8*(n+2)

        #state = self.handler.recv(data_size) # Get state vetor 

            self.client_handler.send(action_buff, self.consock) # Send to socket in UE5
            state = self.client_handler.recv(data_size, self.consock)
            
        else :
            t_obs = self.observation_space.sample()
            t_reward = 0
            t_terminated = False
            state = []
            state = np.append(t_obs,t_reward)
            state = np.append(state, t_terminated)
            
           
        obs = state[0:n]
        reward = state[n]
        terminated = bool(state[n+1])
        self.obs = obs 
        # 4. Rewards System
        # For each frame within the termination limits, 
        ##reward = self.counter
        
        

        state = []
        # Additional metadata [ignore ]
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def connect_socket(self):
        """
            Connect Socket 
            ---
            When each env is called with foreach_env, this method will be invoked 
            in order to create and connect the individual socket for each instance. 

            1. Changes the port based on the env id. 
            2. Get the socket instance 
            3. Connect to given address (self.ip, cutom_socket)

        """
        self.client_handler.set_port(self.port + self.ID)
        self.consock = self.client_handler.set_socket() # Linkea el socket al handler 
        self.client_handler.connect(self.consock) # Intenta conectarse 
    
    def get_socket(self):
        return self.consock

    def set_socket(self, sock):
        """
        Set Socket

        """
        self.consock = sock
        
    def reset(self, *, seed=None, options=None):

        if self.dummy_action is None:
            print("FIRST RESET")
            self.dummy_action = self.action_space.sample()
            obs, self.reward, self.done, self.truncated, self.info = self.step(self.dummy_action)
        else:
            print('[RESETTING]')
            obs = np.asarray(self.obs, dtype=self.observation_space.dtype)

        self.reset_count = self.reset_count+1

        return obs, {}

    def get_multiagent_state_dict(received_vector: np.array): 
        state = {}

        return 
    def check_termination(self, obs):
        """
            Check Termination 
            ---
            Verify if the episode must continue under a set of conditions 

        """
        terminated = False

        return terminated

    def get_amount_obs(self):
        return self.observation_space.shape[0]
    
    def which_observation(self): 
        """
            Which Obs
            ---
            Print description of the observation space in the environment. 
            Amount of elements per observation and fundamental structure. 
        """
        print(f"action dim: {self.observation_space}")

    def which_action(self): 
        """
            Which Obs
            ---
            Print description of the observation space in the environment. 
            Amount of elements per observation and fundamental structure. 
        """
        print(f"action dim: {self.action_space_dim}")

    def summary(self) -> None: 
        """
            Summary
            ---
            Prints the basic structure of the given space. Observation and 
            Actions structures, environment nameid and metadata 
        """
        TEXT_OFFSET = 80
        
        print("")
        print("THIS IS A AUTOGENERATED SUMMARY FOR THE CUSTOM BRIDGE ENV")
        print("--- ")

        print("\nEnvironment: \"{}\"".format(self.name))
        print("This is a gymnasium environment that connects to UE5 via TCP/IP")
        print(f"ip address: {self.ip}\t\t port: {self.port}")
        print("_" * TEXT_OFFSET)
        print("Feature\t\t\t\tType\t\t\t\tShape")
        print("=" * TEXT_OFFSET)

        print("Observation\t\t\t{}\t\t\t\t{}".format(
            self.get_space_type_instance(self.observation_space), # self.get_space_type_id(self.observation_config["type"]),
            self.observation_space.shape)
            )
        
        print("Action\t\t\t\t{}\t\t\t\t{}".format(
              self.get_space_type_instance(self.action_space), # self.get_space_type_id(self.observation_config["type"]),
             self.action_space.shape)
            )
        print("_"* TEXT_OFFSET)        
        print("")

    
    def get_space_type_instance(self, is_instance_object):
        """
            Get space type from instance. 

            Take the space instance. Depending on its class, returns a string 
            identificator the representes the given space. 

            Arguments 
            ---
            - is_instance_object: instance of the space class to compare 

            Returns
            ---
            - String identificator of the space to convert. 

        """
        if isinstance(is_instance_object, BridgeSpaces.Box):
            return "Box Space"
        elif isinstance(is_instance_object, BridgeSpaces.Discrete):
            return "Discrete Space"
    
        return "N/A"

    def create_handler(self): 
        self.handler = ClientHandler(self.ip, self.port) # Create a Handler 
        self.has_handler = True 

    def validate_handler(self): 
        return self.has_handler

        
    def connect(self):
        if self.has_handler:
            self.has_connection = self.handler.connect()
        
        
    def shutdown_server(self):
        """
            Shutdown Server 
            ---

        """
        self.handler.close()

    def set_ID(self, ID):
        self.ID = ID
    
    def get_ID(self):
        return self.ID
        
    
class MultiAgentBridgeEnv(BridgeEnv, MultiAgentEnv):
    """
        MultiAgentBridgeEnv
    """
    def __init__(self, 
                 name: str, 
                 ip: str, 
                 port: int, 
                 config: dict, 
                 first_connection = False, 
                 validation = False, 
                 multiagent = False, 
                 #Paralell
                 ID = int,
                 isTuner = False):
        
        # gui 
        self.ID = ID
        self.client_handler = ClientHandler(ip, port + ID)

        # Connection stage 
        ## Worker will wait until de client handler connect to the UE5 instance Socket Server (SS)
    

        self.ip = ip # IP Address for IP Connection 
        self.port = port 
        #self.bridge = env_bridge

        if not name:
            print("error")
            raise ValueError("no environment name defined. Please define a name for tour environment on the BridgeEnv constructor.")
        self.name = name # environment name for later id 

        # get all agents names 
        self.agent_names = [] # get all agents names 
        self.observations = {}
        self.actions = {}
        self.can_sees = {}
        self.obs_order = {}
        self.reset_count = 0

        
        self.agents_names = list(config.keys())
        print(f"Agents: {self.agents_names}")

        # print("Multiagent params --")
        # print(" - Agent names: {}".format(self.agent_names))
        # print(" - Observations: {}".format(self.observations))
        # print(" - Actions: {}".format(self.actions))
        # print(" - Can see: {}".format(self.can_sees))
        # print(" - Observation order: {}".format(self.obs_order))

        print(" ")

        self.has_handler = False # ClientHandler to begin with connection
        self.has_connection = False

        self.validation = validation
        self.multiagent = multiagent 

        # Rllib metadata
        self.obs_space_dict = self.get_dict_template()
        self.act_space_dict = self.get_dict_template()

        #Dictionary for obs and action spaces for each agent
        for idx, agent in enumerate(self.agents_names):
            self.obs_space_dict[agent] = config[self.agents_names[idx]]['observation']
            self.act_space_dict[agent] = config[self.agents_names[idx]]['action']  

        self.observation_space = config[self.agents_names[0]]['observation']
        self.action_space = config[self.agents_names[0]]['action']

        self.config = config # configuracion de entorno
        self.heads_reference = self.get_dict_template() # where the state vector begins 

        for idx, agent in enumerate(self.agents_names):
            print(self.config[agent])
            self.heads_reference[agent] = 1 if idx == 0 else self.heads_reference[prev_agent] + 3 + self.config[agent]['can_show']
            prev_agent = agent

      

        self.has_connection = True
        self.has_handler = True

        self.obs_dict = self.get_dict_template()
        self.dummy_action = self.get_dict_template()
        if isTuner:
            self.connect_socket()
        

    def get_amount_agents(self) -> int: 
        """
            Get Amount Agents
            ---
            Returns de amount of agents in the multiagent environment.

            @returns amount of agent names (int)
        """
        return len(self.agents_names)

    def validate_actions_dict(self, actions: dict) -> bool:
        """
        Validate Actions Dict
        ---
        Args:
            actions (dict): [description]

        Returns:
            bool: [description]
        """
        return len(actions) == self.get_amount_agents() 
    
    def to_byte(self, byte):
        """
            To byte
            ---
            Convert byte to bits for buffer sned 
        """
        return 8 * byte


    def get_dict_template(self):
        """
            get dict template 
            ---
            Get an agent-name dictionary for using as structure to 
            send to RLLib.
        """
        agent_dict_template = {}
        for agent in self.agents_names:
            agent_dict_template[agent] = ""

        return agent_dict_template
        
    def connect_socket(self):
        """
            Connect Socket 
            ---
            When each env is called with foreach_env, this method will be invoked 
            in order to create and connect the individual socket for each instance. 

            1. Changes the port based on the env id. 
            2. Get the socket instance 
            3. Connect to given address (self.ip, cutom_socket)

        """
        self.client_handler.set_port(self.port + self.ID)
        self.consock = self.client_handler.set_socket() # Linkea el socket al handler 
        self.client_handler.connect(self.consock) # Intenta conectarse 
    
    def step(self, actions: dict) -> None:
        """
        Step 
        """

        action2send = []
        # Will access dict to get actions 
        for action in actions:
            action2send.append(actions[action])


        action=np.array(action2send,dtype=np.single)

        # Validación 

        if isinstance(action, list): 
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            pass
        else:
            assert "No valid action type. Only supports <list> or <numpy.ndarray> given %s" % (type(action))
        
        obs = self.obs_dict 

        
        #print('[ACTION]', end=" ")
        #print(action)

        total_obs_size = 0 # sizes 
        total_obs = []
        agents = self.observations.keys() # agents names

        can_sees_total = []
        order_observations = []

        self.dummy_obs = self.get_dict_template()
        self.dummy_dones = self.get_dict_template()
        self.dummy_reward = self.get_dict_template()
        self.dummy_truncated = self.get_dict_template()
        n_obs = sum([self.config[agent]['can_show'] for agent in self.config])
        # estructura:   (id + obs + reward + done) * agente 
        data_size = self.to_byte(n_obs + self.get_amount_agents() * 3) # bytes from read 

        if self.reset_count > 0:
            #act_2_send = np.insert(action, 0, self.ID)

            #send action to UE5
            self.client_handler.send(action, self.consock) # Send to socket in UE5

            #receive state vector from UE5
            state = self.client_handler.recv(data_size, self.consock)
            
                
            
            #print(f"[STATE]:{state}")
        else:    
            #If is the first reset, get random data
            
            for agent in self.agents_names:
                self.dummy_obs[agent] = self.obs_space_dict[agent].sample()
                self.dummy_dones[agent] = False
                self.dummy_reward[agent] = 0
                self.dummy_truncated[agent] = False 
            
        obs_dict = self.get_dict_template() # from agents names 
        reward_dict = self.get_dict_template() # from agents names 
        done_dict = self.get_dict_template() # from agents names 
        truncated_dict = self.get_dict_template() # from agents names 

        
        acum = 0
        all_done = True


        # Read the specific observations vector 
        # 
        #  id = 1 
        #  obs = total_obs[idx]

        head = 0
        heads = []


        ## 3. PROCESS DATA: Get the state vector and process it 

        if self.reset_count > 0:
            for agent in self.agents_names:
                # amount of observations in the agent 
                obs_dict_arr = []
                for observation_check_agent in list(self.config[agent]['obs_order']):
                    for idx_observation_check_agent in self.config[agent]['obs_order'][observation_check_agent]:            
                        obs_dict_arr.append(state[self.heads_reference[observation_check_agent]] + idx_observation_check_agent)
                
                reward_dict[agent] = state[self.heads_reference[agent] + self.config[agent]['can_show'] ] 
                done_dict[agent] =  bool(state[self.heads_reference[agent] + self.config[agent]['can_show'] + 1])
                obs_dict[agent] = np.asarray(obs_dict_arr, dtype=self.obs_space_dict[agent].dtype) # Add all states needed for agent 
                truncated_dict[agent] = False
                if agent in done_dict:
                    all_done = all_done and done_dict[agent]

        else:
            obs_dict = self.dummy_obs
            reward_dict = self.dummy_reward
            done_dict = self.dummy_dones
            truncated_dict = self.dummy_truncated
            
            

        done_dict["__all__"] = bool(all_done )
        truncated_dict["__all__"] = False 

        self.obs_dict = obs_dict

        info = {}

        return obs_dict, reward_dict, done_dict, truncated_dict, info
    

    def reset(self, *, seed=None, options=None):
        """
            Reset
            ---

        """
        if len(str(self.dummy_action[self.agents_names[0]])) == 0:
            print("FIRST RESET")
            for agent in self.agents_names:
                self.dummy_action[agent] = self.act_space_dict[agent].sample()
            obs_dict, self.reward_dict, self.done_dict, self.truncated_dict, self.info = self.step(self.dummy_action)
        else:
            print('[RESETTING]')
            obs_dict = self.get_dict_template() # from agents names 
            for agent in obs_dict:
                obs_dict[agent] = np.asarray(self.obs_dict[agent], dtype=self.obs_space_dict[agent].dtype)
            #print("OBS DICT: ", obs_dict)
        self.reset_count = self.reset_count+1
        return obs_dict, {}





