from pandas import read_csv
import numpy as np
from copy import deepcopy

class Environment():
    """
    A class representing the environment in which agents operate, including travel times, 
    number of locations, and other relevant settings.

    Attributes:
        travel_time (numpy.ndarray): Matrix of travel times between locations.
        num_locations (int): Number of locations in the environment.
        num_days_trained (int): Number of days the environment has been trained for.
        num_agents (int): Number of agents operating in the environment.
        start_epoch (int): The start time of the environment's operation.
        stop_epoch (int): The end time of the environment's operation.
        epoch_length (int): Length of each time epoch.
        car_capacity (int): Capacity of each car/agent in terms of number of orders.
        current_time (int): Current time in the environment.
        shift_length (int): Length of each agent's shift.
        M (int): Maximum value used for certain calculations in the environment.
    """

    def __init__(self, num_agents, epoch_length, car_cap, data, shift_length, road_speed):
        """
        Constructor for Environment class.

        Parameters:
            num_agents (int): Number of agents operating in the environment.
            epoch_length (int): Length of each time epoch.
            car_cap (int): Capacity of each car/agent in terms of number of orders.
            data (str): Name of the dataset to load travel times from.
            shift_length (int): Length of each agent's shift.
            road_speed (float): Average road speed.
        """
        self.travel_time = read_csv(f'../data/datasets/{data}/travel_time_{int(road_speed)}.csv', header=None).values
        self.num_locations = len(self.travel_time)
        self.num_days_trained = 0
        self.num_agents = num_agents
        self.start_epoch = 0
        self.stop_epoch = 1440
        self.epoch_length = epoch_length
        self.car_capacity = car_cap
        self.current_time = 0
        self.shift_length = shift_length
        self.M = self.get_M_value()

    def get_M_value(self):
        """
        Calculates the M value, used for various computations in the environment.

        Returns:
            int: The calculated M value.
        """
        leave = max(self.travel_time[0])
        back = max(np.transpose(self.travel_time)[0])
        rem_travel = self.travel_time[1:, 1:]
        sorted_indices = np.argsort(rem_travel, axis=None)[::-1]
        highest_values = rem_travel.flat[sorted_indices[:self.car_capacity]]
        return (leave + sum(highest_values) + back) + 1

    def get_travel_time(self, source, destination):
        """
        Retrieves the travel time between two locations.

        Parameters:
            source (int): The source location index.
            destination (int): The destination location index.

        Returns:
            int: The travel time between the source and destination.
        """
        return self.travel_time[source, destination]

    def _check_break_status(self, agent):
        """
        Checks if an agent is currently on a break.

        Parameters:
            agent (Agent): The agent to check the break status for.

        Returns:
            bool: True if the agent is on break, False otherwise.
        """
        return (self.current_time < agent.shift_start) or (self.current_time >= agent.shift_end)

    def _get_ordering_return_time(self, orders):
        """
        Calculates the return time for a given set of orders.

        Parameters:
            orders (list[Order]): List of orders.

        Returns:
            int: The time at which the agent will return after completing the orders.
        """
        ordering = [order.destination for order in orders]
        unique_ordering = list(dict.fromkeys(ordering))
        full_ordering = [0] + unique_ordering + [0]
        location_arrival_times = {}
        time = 0
        for index, location in enumerate(full_ordering[:-1]):
            next_location = full_ordering[index + 1]
            time += self.travel_time[location][next_location]
            location_arrival_times[next_location] = time
        return location_arrival_times[0]

	def simulate_vehicle_motion(self, agent):
	    """
	    Simulates the motion of a vehicle or agent within the environment. This method updates 
	    the agent's status based on its current position (at the warehouse or delivering orders), 
	    time until return, and the orders it has to pick up.

	    Parameters:
	        agent (Agent): The agent for which to simulate motion.
	    """
		# If agent is not on their break
		if not self._check_break_status(agent):
			# If the agent is currently at the warehouse at the beginning of decision epoch
			if agent.time_until_return == 0:
				# If the agent has orders it needs to deliver (i.e. the action assigned to it was to deliver some new orders)
				if len(agent.orders_to_pickup) > 0:
					agent.time_until_return = max((self._get_ordering_return_time(agent.orders_to_pickup) - self.epoch_length), 0)
					agent.orders_to_pickup = []
				else:
					assert not len(agent.orders_to_pickup)
					assert not agent.time_until_return
			# If the agent is not at the warehouse and is out delivering orders
			else:
				# If the agent wont make it back to the warehouse before the end of the decision epoch
				if agent.time_until_return > self.epoch_length:
					agent.time_until_return -= self.epoch_length
					assert agent.time_until_return > 0
				# If the agent will make it back to the warehouse before the end of the decision epoch
				else:
					# If the agent has more orders it needs to pick up and deliver when it gets back to the warehouse
					if len(agent.orders_to_pickup) > 0:
						agent.time_until_return = max(self._get_ordering_return_time(agent.orders_to_pickup) - (self.epoch_length - agent.time_until_return),0)
						agent.orders_to_pickup = []
					# If the agent has no more orders to pickup and deliver when it gets back to the warehouse
					else:
						assert not len(agent.orders_to_pickup)
						agent.time_until_return = 0
		new_time = self.current_time + self.epoch_length
		# Check if in the next time-step the agent should be on their break
		if (new_time < agent.shift_start) and (new_time >= agent.shift_end):
			assert not agent.time_until_return
			assert not len(agent.orders_to_pickup)





					