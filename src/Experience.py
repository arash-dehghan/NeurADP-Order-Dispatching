class Experience(object):
    def __init__(self, agents, feasible_actions_all_agents, current_order_ids, time, num_requests, 
                 num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles, 
                 num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap):

        # Initialize instance variables
        self.agents = agents
        self.feasible_actions_all_agents = feasible_actions_all_agents
        self.current_order_ids = current_order_ids
        self.time = time
        self.num_requests = num_requests
        self.num_agents_on_break = num_agents_on_break
        self.num_agents_at_warehouse = num_agents_at_warehouse
        self.avg_capacity_vehicles = avg_capacity_vehicles
        self.num_past_requests = num_past_requests
        self.num_past_at_warehouse = num_past_at_warehouse
        self.num_past_on_break = num_past_on_break
        self.num_past_cap = num_past_cap

        self.formatting_next = None
        self.formatting_current = None
        self.actions_shape = (len(self.agents), [len(row) for row in self.feasible_actions_all_agents])

        # Assertions to ensure the environment and agents are correctly set
        assert len(agents) == self.envt.num_agents
        assert len(feasible_actions_all_agents) == self.envt.num_agents
