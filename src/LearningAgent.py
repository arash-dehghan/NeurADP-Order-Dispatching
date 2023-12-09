class LearningAgent(object):
    """
    A class representing a learning agent, such as a delivery vehicle or a courier, 
    in a simulation environment. The agent has a defined shift, during which it 
    can pick up and deliver orders.

    Attributes:
        id (int): Unique identifier for the agent.
        shift_start (int): Start time of the agent's shift in minutes.
        shift_length (int): Length of the agent's shift in minutes.
        shift_end (int): End time of the agent's shift in minutes.
        time_until_return (int): Time until the agent returns to the base/warehouse.
        orders_to_pickup (list[Order]): List of orders the agent needs to pick up.
        state (list): Current state of the agent.
        state_str (str): String representation of the agent's current state.
    """

    def __init__(self, agent_id, shift_length, shift_start, current_time=0):
        """
        Constructor for the LearningAgent class.

        Parameters:
            agent_id (int): Unique identifier for the agent.
            shift_length (int): Length of the agent's shift in hours.
            shift_start (int): Start time of the agent's shift in hours.
            current_time (int, optional): The current time in the environment. Defaults to 0.
        """
        self.id = agent_id
        self.shift_start = shift_start * 60  # Convert hours to minutes
        self.shift_length = shift_length * 60  # Convert hours to minutes
        self.shift_end = self.shift_start + self.shift_length
        self.time_until_return = 0  # Time until the agent returns to the warehouse
        self.orders_to_pickup = []  # Orders assigned to the agent for pickup
        self.update_state(current_time)

    def __str__(self):
        """
        String representation of the LearningAgent.

        Returns:
            str: A string describing the agent's ID, shift timings, time until return, and orders to pick up.
        """
        return f'Agent {self.id} [{self.shift_start}, {self.shift_end}, {self.time_until_return}, {len(self.orders_to_pickup)}]'

    def __repr__(self):
        """
        Official string representation of the LearningAgent, used for debugging.

        Returns:
            str: String representation of the LearningAgent object.
        """
        return str(self)

    def update_state(self, current_time):
        """
        Updates the state of the agent based on the current time.

        Parameters:
            current_time (int): The current time in the environment.
        """
        # Update the state with the shift timings, time until return, current orders, and current time
        self.state = [self.shift_start, self.shift_end, self.time_until_return, len(self.orders_to_pickup), current_time]
        self.state_str = str(self.state)
