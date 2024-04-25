from copy import deepcopy
import time
import pickle
import numpy as np
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
from os.path import isfile, isdir

from RequestOrder import RequestOrder
from keras.models import Model, load_model, clone_model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Masking, Concatenate, Flatten, Bidirectional
from keras.backend import function as keras_function
from keras.initializers import Constant

class NeurADP():
	def __init__(self, envt, delay_allowed, trainable, emb_file_name, load_model_loc=''):
		"""
		Neural Network-based Approximate Dynamic Programming (NeurADP) class for learning and decision making.

		Attributes:
			envt (Environment): The simulation environment.
			delay_allowed (int): The maximum delay allowed for an order.
			trainable (bool): Indicates if the model is trainable.
			emb_file_name (str): File name for embedding weights.
			model (keras.models.Model): The neural network model for the value function.
			target_model (keras.models.Model): A target network model for stable training.
			replay_buffer (ReplayBuffer): The buffer storing experiences for training.
			gamma (float): Discount factor for future rewards.
			batch_size_fit (int): Batch size for fitting the model.
			batch_size_predict (int): Batch size for predictions.
			target_update_tau (float): Rate for updating the target model.
			num_min_train_samples (int): Minimum number of samples in buffer to start training.
			num_samples (int): Number of samples to use in each training iteration.
			M (int): A constant used in scoring actions.
		"""
		self.envt = envt
		self.delay_allowed = delay_allowed
		self.trainable = trainable
		self.emb_file_name = emb_file_name

		self.gamma = 0.9
		self.batch_size_fit = 32 # Number of samples per batch to use during fitting
		self.batch_size_predict = 32 # Number of samples per batch to use during prediction
		self.target_update_tau = 0.1
		self.num_min_train_samples = 1000 # Minimum size of replay buffer needed to begin sampling
		self.num_samples = 50
		self.M = self.envt.M

		# Get Replay Buffer
		min_len_replay_buffer = 1e6 / self.envt.num_agents
		epochs_in_episode = (self.envt.stop_epoch - self.envt.start_epoch) / self.envt.epoch_length
		len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
		self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

		# Get NN Model
		self.model = load_model(load_model_loc) if load_model_loc else self._init_NN()
		
		# Define Loss and Compile
		self.model.compile(optimizer='adam', loss='mean_squared_error')

		# Get target-NN
		self.target_model = clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())

		# Define soft-update function for target_model_update
		self.update_target_model = self._soft_update_function(self.target_model, self.model)
		

	def _init_NN(self):
		"""
		Initializes the neural network model used for the value function.

		Returns:
			keras.models.Model: The initialized neural network model.
		"""
		# Depending on whether the model is trainable, use trainable embeddings or pre-loaded embeddings
		if self.trainable:
			# Create a trainable embedding layer for locations
			location_embed = Embedding(output_dim=100, input_dim=self.envt.num_locations + 1, 
									  mask_zero=True, name='location_embedding', trainable=True)
		else:
			# Load pre-trained embeddings for locations
			weights_file = f'../data/generations/{self.emb_file_name}/{self.emb_file_name}.embedding'
			weights = pickle.load(open(weights_file, 'rb'))
			location_embed = Embedding(output_dim=weights[0].shape[1], input_dim=self.envt.num_locations + 1, 
									  mask_zero=True, name='location_embedding', 
									  embeddings_initializer=Constant(weights[0]), trainable=False)

		# Input layer for locations
		locations_input = Input(shape=(self.envt.car_capacity,), dtype='int32', name='locations_input')
		path_location_embed = location_embed(locations_input)

		# Input layer for delays
		delays_input = Input(shape=(self.envt.car_capacity, 1), name='delays_input')
		delay_masked = Masking(mask_value=-1)(delays_input)
		
		# Concatenate location and delay embeddings and pass through an LSTM layer
		path_input = Concatenate()([path_location_embed, delay_masked])
		path_embed = LSTM(200, go_backwards=True)(path_input)

		# Input and embedding layer for the current time
		current_time_input = Input(shape=(1,), name='current_time_input')
		current_time_embed = Dense(100, activation='elu', name='time_embedding')(current_time_input)

		# Additional input layers for various features
		num_requests_input = Input(shape=(1,), name='num_requests_input')
		num_on_break_input = Input(shape=(1,), name='num_on_break_input')
		num_at_warehouse_input = Input(shape=(1,), name='num_at_warehouse_input')
		shift_time_input = Input(shape=(1,), name='shift_time_input')
		return_time_input = Input(shape=(1,), name='return_time_input')
		capacity_input = Input(shape=(1,), name='capacity_input')
		avg_capacity_vehicles_input = Input(shape=(1,), name='avg_capacity_vehicles_input')

		# Concatenation of all embeddings and input features
		state_embed = Concatenate()([path_embed, current_time_embed, num_requests_input, 
									 num_on_break_input, num_at_warehouse_input, shift_time_input, 
									 return_time_input, capacity_input, avg_capacity_vehicles_input])

		# Dense layers for further processing
		state_embed = Dense(300, activation='elu', name='state_embed_1')(state_embed)
		state_embed = Dense(300, activation='elu', name='state_embed_2')(state_embed)

		# Output layer to predict the value function
		output = Dense(1, name='output')(state_embed)

		# Creating the model and specifying inputs and outputs
		model = Model(inputs=[current_time_input, num_requests_input, num_on_break_input, 
							  num_at_warehouse_input, shift_time_input, return_time_input, 
							  locations_input, capacity_input, avg_capacity_vehicles_input, 
							  delays_input], outputs=output)

		return model


	def _soft_update_function(self, target_model, source_model):
		"""
		Creates a function to softly update the weights of the target model using the weights from the source model.

		Parameters:
			target_model (keras.models.Model): The target neural network model whose weights are to be updated.
			source_model (keras.models.Model): The source neural network model providing the updated weights.

		Returns:
			function: A Keras function to perform the soft update.
		"""
		# Extract trainable weights from both target and source models
		target_weights = target_model.trainable_weights
		source_weights = source_model.trainable_weights

		# Prepare a list of update operations
		updates = []
		for target_weight, source_weight in zip(target_weights, source_weights):
			# Calculate the updated weight using a weighted average
			# self.target_update_tau determines the mixing ratio
			updated_weight = self.target_update_tau * source_weight + (1. - self.target_update_tau) * target_weight
			# Append the update operation to the list
			updates.append((target_weight, updated_weight))

		# Create and return a Keras function to perform these updates
		return keras_function([], [], updates=updates)


	def get_value(self, experiences, network=None):
		"""
		Evaluates and scores the given experiences using the neural network model.

		Parameters:
			experiences (list[Experience]): A list of experiences to be evaluated.
			network (keras.models.Model, optional): An alternative neural network model to use for prediction. 
													If None, the primary model of the instance is used.

		Returns:
			list: A list of scored actions for all agents in each experience.
		"""
		# Format experiences into a suitable format for input into the neural network
		action_inputs_all_agents, shape_info = self._format_experiences(experiences, is_current=False)

		# Score experiences using the appropriate neural network model
		if network is None:
			# Use the primary model of the instance for prediction
			expected_future_values_all_agents = self.model.predict(action_inputs_all_agents, batch_size=self.batch_size_predict)
		else:
			# Use the provided alternative network for prediction
			expected_future_values_all_agents = network.predict(action_inputs_all_agents, batch_size=self.batch_size_predict)

		# Reconstruct the output of the neural network back into the format of a 2D list
		# The list is structured for each agent and their corresponding actions
		expected_future_values_all_agents = self._reconstruct_NN_output(expected_future_values_all_agents, shape_info)

		# Extract the feasible actions from the experiences
		feasible_actions_all_agents = [feasible_actions for experience in experiences for feasible_actions in experience.feasible_actions_all_agents]

		# Initialize a list to store the scores for each agent's feasible actions
		scored_actions_all_agents = []
		for expected_future_values, feasible_actions in zip(expected_future_values_all_agents, feasible_actions_all_agents):
			# Score each action using the predicted future values
			scored_actions = [self._get_score(action, value) for action, value in zip(feasible_actions, expected_future_values)]
			scored_actions_all_agents.append(scored_actions)

		# Return the list of scored actions for all agents across all experiences
		return scored_actions_all_agents


	def _format_experiences(self, experiences, is_current):
		"""
		Formats experiences into a structure suitable for input into the neural network model.

		Parameters:
			experiences (list[Experience]): A list of experiences to be formatted.
			is_current (bool): Flag indicating whether to use the current state of the agent or the next state.

		Returns:
			np.array: An array of formatted experiences ready for neural network input.
		"""
		action_inputs_all_agents = None
		for experience in experiences:
			# Check if the experience has already been formatted and stored
			if not (self.__class__.__name__ in experience.representation):
				# Format and store the post-decision state data for the experience
				experience.representation[self.__class__.__name__] = self._get_input_batch_next_state(experience)

			# Determine which batch input to use based on whether current or next state is needed
			if is_current:
				# Format the current state of each agent in the experience
				batch_input = self._format_input_batch([[agent] for agent in experience.agents], 
													   experience.time, experience.num_past_requests, 
													   experience.num_past_on_break, experience.num_past_at_warehouse, 
													   experience.num_past_cap)
			else:
				# Use the stored formatted post-decision state data
				batch_input = deepcopy(experience.representation[self.__class__.__name__])

			# Combine the batch inputs from all experiences
			if action_inputs_all_agents is None:
				action_inputs_all_agents = batch_input

		assert action_inputs_all_agents is not None

		# Flatten the combined NN input to be ready for input into the neural network
		return self._flatten_NN_input(action_inputs_all_agents)

	def _get_input_batch_next_state(self, experience):
		"""
		Generates a batch of inputs representing the next state of each agent after taking actions.

		Parameters:
			experience (Experience): The current experience containing agents and their feasible actions.

		Returns:
			dict: A dictionary of inputs formatted for the neural network, representing the next state of agents.
		"""
		# List to store post-action states for all agents
		all_agents_post_actions = []

		# Iterate over each agent and their feasible actions
		for agent, feasible_actions in zip(experience.agents, experience.feasible_actions_all_agents):
			agents_post_actions = []

			# Iterate over feasible actions for the current agent
			for action in feasible_actions:
				# Create a new agent instance for each action to simulate its outcome
				agent_next_time = deepcopy(agent)

				# If the action involves picking up orders, update the agent's state accordingly
				if len(action[0]) != 0:
					agent_next_time.orders_to_pickup = deepcopy(action[0])

				# Simulate the agent's motion to get the next state after taking the action
				self.envt.simulate_vehicle_motion(agent_next_time)

				# Add the post-action state of the agent to the list
				agents_post_actions.append(agent_next_time)

			# Ensure the number of post-action states matches the number of feasible actions
			assert len(feasible_actions) == len(agents_post_actions)

			# Append the list of post-action states for the current agent to the overall list
			all_agents_post_actions.append(agents_post_actions)

		# Calculate the time for the next state
		next_time = experience.time + self.envt.epoch_length

		# Format the batch of next-state inputs for all agents
		return self._format_input_batch(all_agents_post_actions, next_time, experience.num_requests, 
										experience.num_agents_on_break, experience.num_agents_at_warehouse, 
										experience.avg_capacity_vehicles)
	
	def _format_input_batch(self, all_agents_post_actions, current_time, num_requests, num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles):
		"""
		Formats a batch of inputs for the neural network based on the post-action states of all agents.

		Parameters:
			all_agents_post_actions (list[list[Agent]]): A list of lists, where each sublist represents the post-action states of a single agent.
			current_time (int): The current time in the simulation.
			num_requests (int): The number of requests at the current time.
			num_agents_on_break (int): The number of agents currently on a break.
			num_agents_at_warehouse (int): The number of agents currently at the warehouse.
			avg_capacity_vehicles (float): The average capacity of vehicles.

		Returns:
			dict: A dictionary with formatted neural network inputs.
		"""
		# Initialize a dictionary to store neural network inputs
		inp = {
			"current_time_input": [],
			"shift_time_input": [],
			"return_time_input": [],
			"num_requests_input": [],
			"locations_input": [],
			"capacity_input": [],
			"delays_input": [],
			"num_on_break_input": [],
			"num_at_warehouse_input": [],
			"avg_capacity_vehicles_input": []
		}

		# Process each agent's post-action states
		for agent_post_actions in all_agents_post_actions:
			# Initialize lists for each type of input
			current_times, shift_time_inputs, return_time_inputs, num_requests_inputs, locations_inputs, capacity_inputs, delays_inputs, num_on_break_inputs, num_at_warehouse_inputs, avg_capacity_vehicles_inputs = ([] for _ in range(10))

			# Format inputs for each post-action state of an agent
			for agent in agent_post_actions:
				current_time_input, shift_time_input, return_time_input, num_requests_input, locations_input, capacity_input, delay_input, num_on_break_input, num_at_warehouse_input, avg_capacity_vehicles_input = self._format_input(
					agent, current_time, num_requests, num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles
				)

				# Append the formatted data to the corresponding lists
				current_times.append(current_time_input)
				shift_time_inputs.append(shift_time_input)
				return_time_inputs.append(return_time_input)
				num_requests_inputs.append(num_requests_input)
				locations_inputs.append(locations_input)
				capacity_inputs.append(capacity_input)
				delays_inputs.append(delay_input)
				num_on_break_inputs.append(num_on_break_input)
				num_at_warehouse_inputs.append(num_at_warehouse_input)
				avg_capacity_vehicles_inputs.append(avg_capacity_vehicles_input)

			# Add the lists of formatted data to the input dictionary
			inp["current_time_input"].append(current_times)
			inp["num_requests_input"].append(num_requests_inputs)
			inp["shift_time_input"].append(shift_time_inputs)
			inp["return_time_input"].append(return_time_inputs)
			inp["locations_input"].append(locations_inputs)
			inp["capacity_input"].append(capacity_inputs)
			inp["delays_input"].append(delays_inputs)
			inp["num_on_break_input"].append(num_on_break_inputs)
			inp["num_at_warehouse_input"].append(num_at_warehouse_inputs)
			inp["avg_capacity_vehicles_input"].append(avg_capacity_vehicles_inputs)

		return inp


	def _format_input(self, agent, current_time, num_requests, num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles):
		"""
		Formats and normalizes input data for a single agent for neural network processing.

		Parameters:
			agent (Agent): The agent whose data is being formatted.
			current_time (int): The current time in the simulation.
			num_requests (int): The number of requests at the current time.
			num_agents_on_break (int): The number of agents currently on a break.
			num_agents_at_warehouse (int): The number of agents currently at the warehouse.
			avg_capacity_vehicles (float): The average capacity of vehicles.

		Returns:
			tuple: A tuple of normalized inputs for the neural network.
		"""
		# Normalizing and formatting agent's locations and delays
		locations_input = self._add_locations(agent.orders_to_pickup)
		delays_input = self._add_delays(agent.orders_to_pickup, current_time)

		# Normalizing current time, number of requests, and agents' statuses
		current_time_input = (current_time - self.envt.start_epoch) / (self.envt.stop_epoch - self.envt.start_epoch)
		num_requests_input = num_requests / 200  # Assuming 200 is a normalization factor
		num_agents_on_break_input = num_agents_on_break / self.envt.num_agents
		num_at_warehouse_input = num_agents_at_warehouse / self.envt.num_agents

		# Normalizing agents' shift time and time until return
		shift_time_input = agent.shift_start / 1440  # Assuming a day has 1440 minutes
		return_time_input = agent.time_until_return / 1440

		# Normalizing agents' current capacity and average capacity of vehicles
		capacity_input = len(agent.orders_to_pickup) / self.envt.car_capacity
		avg_capacity_vehicles_input = avg_capacity_vehicles / self.envt.car_capacity

		# Return the normalized inputs as a tuple
		return current_time_input, shift_time_input, return_time_input, num_requests_input, locations_input, capacity_input, delays_input, num_agents_on_break_input, num_at_warehouse_input, avg_capacity_vehicles_input


	def _add_locations(self, locations):
		"""
		Prepares and formats the locations data for neural network input.

		Parameters:
			locations (list[Location]): A list of locations related to the agent's orders.

		Returns:
			np.array: An array representing the locations formatted for neural network input.
		"""
		# Initialize an array to hold location inputs with zeros
		locations_input = np.zeros(shape=(self.envt.car_capacity,), dtype='int32')

		# Iterate over each location in the provided list
		for loc_id in range(len(locations)):
			# Assign the location (destination + 1 to account for zero-indexing) to the array
			locations_input[loc_id] = locations[loc_id].destination + 1

		return locations_input

	def _add_delays(self, orders, current_time):
		"""
		Prepares and formats the delays data for each order for neural network input.

		Parameters:
			orders (list[Order]): A list of orders associated with the agent.
			current_time (int): The current time in the simulation.

		Returns:
			np.array: An array representing the delays for each order formatted for neural network input.
		"""
		# Initialize an array to hold delay inputs with -1 (indicating no delay/order)
		delays = np.zeros(shape=(self.envt.car_capacity, 1)) - 1

		# Iterate over each order
		for i, order in enumerate(orders):
			# Calculate the normalized delay for the order
			delays[i] = (order.deadline - current_time) / (order.deadline - order.origin_time)

		return delays

	def _flatten_NN_input(self, NN_input):
		"""
		Flattens the neural network input from a dictionary of 2D lists into a dictionary of 1D arrays.

		Parameters:
			NN_input (dict): A dictionary where each key corresponds to a neural network input feature and 
							 each value is a 2D list representing agents and their actions.

		Returns:
			tuple: A tuple containing the flattened input dictionary and the shape information.
		"""
		# Initialize a list to store shape information of the inputs
		shape_info = []

		for key, value in NN_input.items():
			# Store the shape information of the inputs for later reconstruction
			if not shape_info:
				cumulative_sum = 0
				shape_info.append(cumulative_sum)
				for idx, list_el in enumerate(value):
					cumulative_sum += len(list_el)
					shape_info.append(cumulative_sum)

			# Flatten the 2D list into a 1D array for each key in the dictionary
			NN_input[key] = np.array([element for array in value for element in array])

		# Return the dictionary with flattened inputs and the shape information
		return NN_input, shape_info


	def _reconstruct_NN_output(self, NN_output, shape_info):
		"""
		Reconstructs the neural network output back into its original 2D list format.

		Parameters:
			NN_output (np.array): The flattened output from the neural network.
			shape_info (list): Information about the original shape of the output.

		Returns:
			list: A 2D list representing the output for each agent and their corresponding actions.
		"""
		# Flatten the output to a 1D array
		NN_output = NN_output.flatten()

		# Reconstruct the output back to a 2D list structure
		output_as_list = []
		for idx in range(len(shape_info) - 1):
			start_idx = shape_info[idx]
			end_idx = shape_info[idx + 1]
			list_el = NN_output[start_idx:end_idx].tolist()
			output_as_list.append(list_el)

		return output_as_list

	def _get_score(self, action, value):
		"""
		Calculates the score for a given action based on its value.

		Parameters:
			action (tuple): The action taken by the agent.
			value (float): The predicted value of taking the action.

		Returns:
			float: The calculated score for the action.
		"""
		# Calculate the score based on the number of orders in the action, a constant M, and the predicted value
		return (len(action[1]) * self.M - action[2]) + self.gamma * value


	def pair_scores(self, scored_actions_all_agents, agents_matchings):
		"""
		Pairs scores with actions for each agent.

		Parameters:
			scored_actions_all_agents (list): A list of scored actions for all agents.
			agents_matchings (list): A list of actions and their matchings for each agent.

		Returns:
			dict: A dictionary of final pairings of scores with actions for each agent.
		"""
		final_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}
		id_to_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}

		for agent_id, (action_matchings, matching_scores) in enumerate(zip(agents_matchings, scored_actions_all_agents)):
			for action, score in zip(action_matchings, matching_scores):
				action_name = f"o_{'_'.join(map(str, action[1]))}"
				final_pairings[agent_id][action_name] = score
				id_to_pairings[agent_id][action_name] = action

		return final_pairings, id_to_pairings


	def remember(self, experience):
		"""
		Adds an experience to the replay buffer.

		Parameters:
			experience (Experience): The experience to be added to the buffer.
		"""
		self.replay_buffer.add(experience)


	def update(self, central_agent):
		"""
		Updates the neural network model based on experiences sampled from the replay buffer.

		Parameters:
			central_agent (CentralAgent): The central agent that provides additional functionality for decision-making.
		"""
		# Check if there are enough samples in the replay buffer to perform an update
		if self.num_min_train_samples > len(self.replay_buffer):
			return

		# Sample experiences from the replay buffer
		if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
			# If using prioritized replay, calculate the beta value for sampling
			beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
			experiences, weights, batch_idxes = self.replay_buffer.sample(self.num_samples, beta)
		else:
			# For simple replay buffer
			experiences = self.replay_buffer.sample(self.num_samples)
			weights = None

		# Iteratively update the policy based on each sampled experience
		for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
			# Apply weights to the experiences if using a prioritized replay buffer
			w = np.array([weights[experience_idx]] * self.envt.num_agents) if weights is not None else None

			# Get the temporal-difference target (TD-target)
			scored_actions_all_agents = self.get_value([experience], network=self.target_model)
			final_pairings, id_to_pairings = self.pair_scores(scored_actions_all_agents, experience.feasible_actions_all_agents)
			_, scores = central_agent.choose_actions(final_pairings, id_to_pairings, len(experience.agents), experience.request_ids, is_training=False)
			value_next_state = [scores[i] for i in range(len(scores))]
			supervised_targets = np.array(value_next_state).reshape((-1, 1))

			# Update the neural network based on the TD-target
			action_inputs_all_agents, _ = self._format_experiences([experience], is_current=True)
			self.model.fit(action_inputs_all_agents, supervised_targets, batch_size=self.batch_size_fit, sample_weight=w, verbose=0)

			# If using a prioritized replay buffer, update the priorities based on the new loss
			if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
				predicted_values = self.model.predict(action_inputs_all_agents, batch_size=self.batch_size_predict)
				loss = np.mean((predicted_values - supervised_targets) ** 2 + 1e-6)
				self.replay_buffer.update_priorities([batch_idx], [loss])

			# Perform a soft update of the target model using the weights from the trained model
			self.update_target_model([])





