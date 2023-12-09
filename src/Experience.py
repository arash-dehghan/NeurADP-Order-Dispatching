class Experience(object):
    """
    A class representing an experience or state within the simulation environment. 
    It encapsulates all relevant information about the current state of agents, 
    their feasible actions, and various metrics about the environment and agents.

    Class Attribute:
        envt (Environment): A class-level attribute representing the environment 
                            shared among all instances of Experience.

    Attributes:
        agents (list[Agent]): List of agents in the environment.
        feasible_actions_all_agents (list[list[tuple]]): List of feasible actions for all agents.
        time (int): Current time in the environment.
        num_requests (int): Number of requests at the current time.
        request_ids (list[int]): List of IDs of current requests.
        num_agents_on_break (int): Number of agents currently on a break.
        num_agents_at_warehouse (int): Number of agents currently at the warehouse.
        avg_capacity_vehicles (float): Average capacity of vehicles in the environment.
        num_past_requests (int): Number of requests in the past.
        num_past_at_warehouse (int): Number of times agents were at the warehouse in the past.
        num_past_on_break (int): Number of times agents were on break in the past.
        num_past_cap (int): Past capacity metrics.
        representation (dict): A dictionary to represent the state.
    """

    envt = None  # Static variable shared by all instances

    def __init__(self, agents, feasible_actions_all_agents, time, num_requests, request_ids, 
                 num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles, 
                 num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap):
        """
        Constructor for the Experience class.

        Parameters:
            agents (list[Agent]): List of agents in the environment.
            feasible_actions_all_agents (list[list[tuple]]): List of feasible actions for all agents.
            time (int): Current time in the environment.
            num_requests (int): Number of requests at the current time.
            request_ids (list[int]): List of IDs of current requests.
            num_agents_on_break (int): Number of agents currently on a break.
            num_agents_at_warehouse (int): Number of agents currently at the warehouse.
            avg_capacity_vehicles (float): Average capacity of vehicles in the environment.
            num_past_requests (int): Number of requests in the past.
            num_past_at_warehouse (int): Number of times agents were at the warehouse in the past.
            num_past_on_break (int): Number of times agents were on break in the past.
            num_past_cap (int): Past capacity metrics.
        """

        # Initialize instance variables
        self.agents = agents
        self.feasible_actions_all_agents = feasible_actions_all_agents
        self.time = time
        self.num_requests = num_requests
        self.request_ids = request_ids
        self.num_agents_on_break = num_agents_on_break
        self.num_agents_at_warehouse = num_agents_at_warehouse
        self.avg_capacity_vehicles = avg_capacity_vehicles
        self.num_past_requests = num_past_requests
        self.num_past_at_warehouse = num_past_at_warehouse
        self.num_past_on_break = num_past_on_break
        self.num_past_cap = num_past_cap

        # Assertions to ensure the environment and agents are correctly set
        assert self.envt is not None
        assert len(agents) == self.envt.num_agents
        assert len(feasible_actions_all_agents) == self.envt.num_agents

        # Initialize the representation of the experience
        self.representation = {}
