from pandas import read_csv
import numpy as np
from copy import deepcopy

class Environment():
	"""
	A class representing the environment in which agents operate, including travel times, 
	number of locations, and other relevant settings.

	Attributes:
		filename (str): Name of file for data generation.
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

	def __init__(self, filename, num_agents, epoch_length, car_cap, data, road_speed, delay_type, breaks_included):
		"""
		Constructor for Environment class.

		Parameters:
			filename (str): Name of file for data generation.
			num_agents (int): Number of agents operating in the environment.
			epoch_length (int): Length of each time epoch.
			car_cap (int): Capacity of each car/agent in terms of number of orders.
			data (str): Name of the dataset to load travel times from.
			shift_length (int): Length of each agent's shift.
			road_speed (float): Average road speed.
		"""
		self.filename = filename
		city, epoch_length, road_speed, _, boundedness, delaytime, seed = self.filename.split('_')
		self.city = city
		self.epoch_length = int(epoch_length)
		self.road_speed = float(road_speed)
		self.boundedness = boundedness
		self.delaytime = float(delaytime)
		self.delay_type = delay_type
		self.breaks_included = breaks_included
		self.seed = int(seed)
		self.travel_time = read_csv(f'../data/datasets/{data}/travel_time_{self.filename}.csv', header=None).values
		self.num_locations = len(self.travel_time)
		self.num_days_trained = 0
		self.num_agents = num_agents
		self.start_epoch = 0
		self.stop_epoch = 1440
		self.car_capacity = car_cap
		self.current_time = 0
		self.shift_length = 6
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

	def _check_break_status(self, agent, time):
		"""
		Checks if an agent is currently on a break.

		Parameters:
			agent (Agent): The agent to check the break status for.

		Returns:
			bool: True if the agent is on break, False otherwise.
		"""
		return (time < agent.shift_start) or (time >= agent.shift_end)

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

	def _get_order_delivery_durations(self, agent, time):
		original_time = time
		orders = deepcopy(agent.orders_to_pickup)
		ordering = [order.destination for order in orders]
		unique_ordering = list(dict.fromkeys(ordering))
		full_ordering = [0] + unique_ordering + [0]
		location_arrival_times = {}
		for index, location in enumerate(full_ordering[:-1]):
			next_location = full_ordering[index + 1]
			time += self.travel_time[location][next_location]
			location_arrival_times[next_location] = time

		# How long it will take to return to warehouse based on this delivery
		delivery_duration = location_arrival_times[0] - original_time

		# Number of orders in delivery
		delivery_number_of_orders = len(orders)

		# Percentage of allowed delay deadline used 
		delivery_deadline_differences = [(location_arrival_times[order.destination] - order.origin_time) / (order.deadline - order.origin_time) for order in orders]
		
		return delivery_duration, delivery_number_of_orders, delivery_deadline_differences


	def simulate_vehicle_motion(self, agent, time):
		delivery_duration, delivery_number_of_orders, delivery_deadline_differences = None, None, []

		if not self._check_break_status(agent, time):
			# If the agent is currently at the warehouse at the beginning of decision epoch
			if agent.time_until_return == 0:
				# If the agent has orders it needs to deliver (i.e. the action assigned to it was to deliver some new orders)
				if len(agent.orders_to_pickup) > 0:
					delivery_duration, delivery_number_of_orders, delivery_deadline_differences = self._get_order_delivery_durations(agent, time)
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
						delivery_duration, delivery_number_of_orders, delivery_deadline_differences = self._get_order_delivery_durations(agent, time + agent.time_until_return)
						agent.time_until_return = max(self._get_ordering_return_time(agent.orders_to_pickup) - (self.epoch_length - agent.time_until_return),0)
						agent.orders_to_pickup = []
					# If the agent has no more orders to pickup and deliver when it gets back to the warehouse
					else:
						assert not len(agent.orders_to_pickup)
						agent.time_until_return = 0

		new_time = time + self.epoch_length
		# Check if in the next time-step the agent should be on their break
		if new_time >= agent.shift_end:
			assert not agent.time_until_return
			assert not len(agent.orders_to_pickup)

		return delivery_duration, delivery_number_of_orders, delivery_deadline_differences

					