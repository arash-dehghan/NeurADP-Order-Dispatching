from itertools import combinations, permutations
import docplex.mp.model as cpx
import cplex
from copy import deepcopy
import numpy as np

class CentralAgent(object):
	"""
	A class representing a central agent that handles the assignment of orders to agents
	(like vehicles or delivery personnel) based on various constraints and objectives.

	Attributes:
		envt (Environment): The environment in which the agents operate.
		num_agents (int): The number of agents available for assigning orders.
		delay_allowed (int): The maximum delay allowed for an order.
		car_capacity (int): The capacity of each car/agent in terms of number of orders.
	"""

	def __init__(self, envt):
		"""
		Constructor for CentralAgent class.

		Parameters:
			envt (Environment): The environment in which the agents operate.
			num_agents (int): The number of agents available for assigning orders.
			delay_allowed (int): The maximum delay allowed for an order.
			car_capacity (int): The capacity of each car/agent in terms of number of orders.
		"""
		self.envt = envt

	def set_deadlines(self, order, order_id):
		"""
		Sets the deadline and ID for an order.

		Parameters:
			order (Order): The order object to set the deadline for.
			order_id (int): The ID to assign to the order.

		Returns:
			Order: The updated order object with set deadline and ID.
		"""
		speed, base_fee, cost_per_km = 20.0, 4.4, 1.8

		if self.envt.boundedness == 'unbounded':
			order.deadline = self.envt.current_time + self.envt.travel_time[0][order.destination] + self.envt.delaytime
		elif self.envt.boundedness == 'bounded':
			order.deadline = (self.envt.current_time + self.envt.delaytime) if self.envt.delay_type == 'getir' else (self.envt.current_time + self.envt.travel_time[0][order.destination] + self.envt.delaytime)      
		order.id = order_id
		order.revenue = base_fee + ((self.envt.travel_time[0][order.destination] / 60) * speed) * cost_per_km
		order.update_state(self.envt.current_time)
		return order

	def _check_break_status(self, agent):
		"""
		Checks if an agent is currently on a break.

		Parameters:
			agent (Agent): The agent to check the break status for.

		Returns:
			bool: True if the agent is on break, False otherwise.
		"""
		return (self.envt.current_time < agent.shift_start) or (self.envt.current_time >= agent.shift_end)

	def _check_at_warehouse_status(self, agent):
		"""
		Checks if an agent is currently at the warehouse.

		Parameters:
			agent (Agent): The agent to check the warehouse status for.

		Returns:
			bool: True if the agent is at the warehouse, False otherwise.
		"""
		return (not agent.time_until_return) and (not self._check_break_status(agent))

	def get_external_infor(self, agents):
		"""
		Gathers external information about agents like their break status and current capacity.

		Parameters:
			agents (list[Agent]): List of agents to gather information from.

		Returns:
			tuple: A tuple containing the number of agents on break, number of agents at the warehouse,
				   and the average capacity of vehicles currently in operation.
		"""
		num_agents_on_break = sum([1 for agent in agents if self._check_break_status(agent)])
		num_agents_at_warehouse = sum([1 for agent in agents if self._check_at_warehouse_status(agent)])
		capacities = [len(agent.orders_to_pickup) for agent in agents if not self._check_break_status(agent)]
		avg_capacity_vehicles = np.mean(capacities) if len(capacities) > 0 else -1
		return num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles

	def get_feasible_actions(self, agents, orders):
		"""
		Determines feasible actions for each agent based on current orders and agent capacity.

		Parameters:
			agents (list[Agent]): The agents to determine actions for.
			orders (list[Order]): The current list of orders.

		Returns:
			list[list[tuple]]: A list of feasible actions for each agent.
		"""
		potential_actions = self._get_potential_actions(orders)
		agent_feasible_actions = []
		for agent in agents:
			agents_actions = [([], [], 0)]
			if not self._check_break_status(agent):
				capacity_available = self.envt.car_capacity - len(agent.orders_to_pickup)
				for cap in range(1, capacity_available + 1):
					for act in potential_actions[cap]:
						ordering = self._check_feasibility(agent, act)
						if len(ordering) > 0:
							return_time = self.envt._get_ordering_return_time(ordering)
							agents_actions.append((ordering, [o.id for o in act], return_time))
			else:
				if self.envt.breaks_included and (self.envt.current_time < agent.shift_end):
					capacity_available = self.envt.car_capacity - len(agent.orders_to_pickup)
					for cap in range(1, capacity_available + 1):
						for act in potential_actions[cap]:
							ordering = self._check_feasibility_on_break(agent, act)
							if len(ordering) > 0:
								return_time = self.envt._get_ordering_return_time(ordering)
								agents_actions.append((ordering, [o.id for o in act], return_time))

			agent_feasible_actions.append(agents_actions)
		return agent_feasible_actions

	def _get_potential_actions(self, orders):
		"""
		Generates potential actions based on the current orders and car capacity.

		Parameters:
			orders (list[Order]): The current list of orders.

		Returns:
			dict: A dictionary with keys as the number of orders and values as a list of order combinations.
		"""
		return {size : [list(action) for action in list(combinations(orders, size))] for size in range(1, self.envt.car_capacity + 1)}

	def _check_feasibility(self, agent, action):
		"""
		Checks the feasibility of an action for a given agent.

		Parameters:
			agent (Agent): The agent to check the feasibility for.
			action (list[Order]): The action to check feasibility of.

		Returns:
			list[Order]: The feasible ordering of orders for the agent.
		"""
		orders = agent.orders_to_pickup + action
		all_destinations = set([order.destination for order in orders])
		best_ordering, best_return_time = [], self.envt.stop_epoch + 1

		for ordering in permutations(all_destinations):
			time = self.envt.current_time + agent.time_until_return
			full_ordering = [0] + list(ordering) + [0]
			location_arrival_times = {}

			for i in range(len(full_ordering) - 1):
				time += self.envt.travel_time[full_ordering[i]][full_ordering[i + 1]]
				location_arrival_times[full_ordering[i + 1]] = time

			if all(order.deadline >= location_arrival_times[order.destination] for order in orders):
				if (location_arrival_times[0] < agent.shift_end):
					if location_arrival_times[0] < best_return_time:
						best_ordering = list(ordering)
						best_return_time = location_arrival_times[0]

		return [order for loc in best_ordering for order in orders if order.destination == loc]

	def _check_feasibility_on_break(self, agent, action):
		orders = agent.orders_to_pickup + action
		all_destinations = set([order.destination for order in orders])
		best_ordering, best_return_time = [], self.envt.stop_epoch + 1

		for ordering in permutations(all_destinations):
			time = agent.shift_start
			full_ordering = [0] + list(ordering) + [0]
			location_arrival_times = {}

			for i in range(len(full_ordering) - 1):
				time += self.envt.travel_time[full_ordering[i]][full_ordering[i + 1]]
				location_arrival_times[full_ordering[i + 1]] = time

			if all(order.deadline >= location_arrival_times[order.destination] for order in orders):
				if (location_arrival_times[0] < agent.shift_end):
					if location_arrival_times[0] < best_return_time:
						best_ordering = list(ordering)
						best_return_time = location_arrival_times[0]

		return [order for loc in best_ordering for order in orders if order.destination == loc]

	def choose_actions(self, agent_action_choices, id_to_pairings, num_agents, request_ids, is_training):
		"""
		Chooses the best actions for each agent using a matching model.

		Parameters:
			agent_action_choices (dict): A dictionary of feasible actions for each agent.
			id_to_pairings (dict): A dictionary mapping agent IDs to action pairings.
			num_agents (int): The number of agents.
			request_ids (list[int]): List of request IDs.
			is_training (bool): Flag indicating whether the model is in training mode.

		Returns:
			tuple: A tuple containing the final actions for each agent and the corresponding scores.
		"""
		# Initialize the optimization model
		model = cpx.Model(name="Matching Model")

		# Define binary variables for each potential agent-action pairing
		x_a_d = {(a,b): model.binary_var(name=f'x_{a}_{b}') for a in agent_action_choices.keys() for b in agent_action_choices[a].keys()}

		# Add constraints to ensure each agent is assigned to at most one action
		flow_driver_conservation_const = {a : model.add_constraint(ct=(model.sum(x_a_d[(a,b)] for b in agent_action_choices[a]) == 1), ctname=f'constraint_a_{a}') for a in range(num_agents)}

		# Add constraints to ensure each order is assigned to at most one agent
		flow_order_conservation_const = {order : model.add_constraint(ct=(model.sum(x_a_d[(a,b)] for a in range(num_agents) for b in agent_action_choices[a] if f'_{order}' in b) <= 1), ctname=f'constraint_o_{order}') for order in request_ids}

		# Define the objective function to maximize the total score of the agent-action pairings
		total_obj = model.sum(x_a_d[(a,b)] * (agent_action_choices[a][b] + self._get_noise(x_a_d[(a,b)], is_training)) for a in agent_action_choices.keys() for b in agent_action_choices[a].keys())
		model.set_objective('max', total_obj)

		# Solve the model
		solution = model.solve()
		assert solution

		# Extract the final actions and their scores based on the optimization results
		final_actions, final_scores = {}, {}
		for a in agent_action_choices.keys():
			for b in agent_action_choices[a].keys():
				if solution.get_value(f'x_{a}_{b}') > 0:
					final_actions[a] = id_to_pairings[a][b]
					final_scores[a] = agent_action_choices[a][b]
		return final_actions, final_scores

	# def _get_noise(self, variable, is_training):
	# 	# return np.random.uniform(0, 2) if is_training else 0
	# 	return np.random.uniform(0, 10) if is_training else 0
	# 	# stdev = 1 + (4000 if variable.get_name()[-1] == '_' else 1000) / ((self.envt.num_days_trained + 1) * self.envt.num_agents)
	# 	# return abs(np.random.normal(0, stdev)) if is_training else 0

	def _get_noise(self, variable, is_training):
		if is_training:
			# Set initial noise level to half of the maximum possible reward
			initial_noise_level = self.envt.car_capacity * 0.5  # For car_capacity=3, this is 1.5
			min_noise_level = self.envt.car_capacity * 0.05     # For car_capacity=3, this is 0.15

			# Decay rate controls how quickly the noise decreases
			decay_rate = 0.01  # Adjust based on desired exploration duration

			# Calculate current noise level using exponential decay
			noise_level = max(
				min_noise_level,
				initial_noise_level * np.exp(-decay_rate * self.envt.num_days_trained)
			)

			# Sample noise from a normal distribution centered at 0
			return np.random.normal(0, noise_level)
		else:
			return 0

	def set_new_paths(self, agents, matchings, time):
		"""
		Sets new paths for agents based on the matching results.

		Parameters:
			agents (list[Agent]): The list of agents to set paths for.
			matchings (dict): The matchings of agents to actions.

		Returns:
			int: The number of orders that were served.
		"""
		served = 0
		durations, sizes, differences = [], [], []
		for i in range(len(agents)):
			agent = agents[i]
			action = matchings[i]
			# If the action is to continue on their path
			if len(action[0]) == 0:
				delivery_duration, delivery_number_of_orders, delivery_deadline_differences = self.envt.simulate_vehicle_motion(agent, time)
				if delivery_duration != None:
					durations.append(delivery_duration)
				if delivery_number_of_orders != None:
					sizes.append(delivery_number_of_orders)
				differences += delivery_deadline_differences
				
			else:
				served += len(action[1])
				agent.orders_to_pickup = deepcopy(action[0])
				delivery_duration, delivery_number_of_orders, delivery_deadline_differences = self.envt.simulate_vehicle_motion(agent, time)
				if delivery_duration != None:
					durations.append(delivery_duration)
				if delivery_number_of_orders != None:
					sizes.append(delivery_number_of_orders)
				differences += delivery_deadline_differences

		return served, durations, sizes, differences




