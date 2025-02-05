import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import numpy as np
from copy import deepcopy
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
import math

class MatchingNN(nn.Module):
	def __init__(self, num_destinations, emb_dim=100):
		super(MatchingNN, self).__init__()

		# Embedding layer
		self.location_embed = nn.Embedding(num_embeddings=num_destinations + 1, embedding_dim=emb_dim, padding_idx=0)

		# LSTM layer
		self.lstm = nn.LSTM(input_size=emb_dim + 1, hidden_size=200, batch_first=True)

		# Embedding layer for current time
		self.time_embedding = nn.Linear(1, 100)

		# Dense layers
		self.state_embed_1 = nn.Linear(200 + 100 + 7, 300)
		self.state_embed_2 = nn.Linear(300, 300)
		self.output_layer = nn.Linear(300, 1)

	def forward(self, locations_input, delays_input, current_time_input,
				num_requests_input, num_on_break_input, num_at_warehouse_input,
				shift_time_input, return_time_input, capacity_input, avg_capacity_vehicles_input):
		# Embedding lookup for locations
		path_location_embed = self.location_embed(locations_input)  # Shape: (batch_size, car_capacity, emb_dim)

		# Concatenate location embeddings with delays
		path_input = torch.cat([path_location_embed, delays_input], dim=2)  # Shape: (batch_size, car_capacity, emb_dim + 1)

		# Reverse the sequence to match 'go_backwards=True' in TensorFlow
		path_input_reversed = torch.flip(path_input, dims=[1])

		# Pass through LSTM
		lstm_out, (hn, cn) = self.lstm(path_input_reversed)
		path_embed = hn[-1]  # Shape: (batch_size, 200)

		# Embed current time
		current_time_embed = F.elu(self.time_embedding(current_time_input))

		# Concatenate all features
		state_embed = torch.cat([path_embed, current_time_embed,
								 num_requests_input,
								 num_on_break_input,
								 num_at_warehouse_input,
								 shift_time_input,
								 return_time_input,
								 capacity_input,
								 avg_capacity_vehicles_input], dim=1)

		# Pass through dense layers
		state_embed = F.elu(self.state_embed_1(state_embed))
		state_embed = F.elu(self.state_embed_2(state_embed))

		# Output layer
		output = self.output_layer(state_embed)
		return output



class NeurADP():
	def __init__(self, envt, central_agent):
		# General information
		self.envt = envt
		self.central_agent = central_agent

		# Matching NN information
		self.model = MatchingNN(self.envt.num_locations)
		self.target = MatchingNN(self.envt.num_locations)
		self.target.load_state_dict(self.model.state_dict())
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.99, patience=500, verbose=False)
		self.gamma = 0.9
		self.target_update_tau = 0.01

		# Move models to device
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.target.to(self.device)

		# Matching Replay Buffer information
		min_len_replay_buffer = 5000
		self.num_min_train_samples = 1000
		self.num_samples = 30
		epochs_in_episode = (self.envt.stop_epoch - self.envt.start_epoch) / self.envt.epoch_length
		len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
		self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

	def soft_update(self):
		with torch.no_grad():
			for target_param, source_param in zip(self.target.parameters(), self.model.parameters()):
				updated_weight = self.target_update_tau * source_param.data + (1 - self.target_update_tau) * target_param.data
				target_param.data.copy_(updated_weight)

	def get_value(self, experience, use_target = False):
		# Format experience
		experience = deepcopy(experience)
		formatted_experience = self._format_experience(experience)

		# Get NN values
		V = self._get_NN_values(formatted_experience, use_target)

		# Get direct rewards
		R = self._get_direct_rewards(experience.feasible_actions_all_agents)

		# Get scores
		raw_scores = list(R + self.gamma * V)
		scores = self._reshape_values(raw_scores, experience.actions_shape)

		return scores

	def _format_experience(self, experience, current_time = False):
		if (experience.formatting_next is None) or (experience.formatting_current is None):
			experience.formatting_next = self._get_next_state_input(experience)
			experience.formatting_current = self._get_current_state_input(experience)
		return deepcopy(experience.formatting_current) if current_time else deepcopy(experience.formatting_next)

	def _get_NN_values(self, formatted_experience, use_target):
		inputs = [x.to(self.device) for x in self._process_experience_into_tensors(formatted_experience)]
		output = self.target(*inputs).cpu().detach().numpy() if use_target else self.model(*inputs).cpu().detach().numpy()
		return np.array([v[0] for v in output])

	def _get_next_state_input(self, experience):
		all_agents_post_actions = []
		for agent, feasible_actions in zip(experience.agents, experience.feasible_actions_all_agents):
			agents_post_actions = []
			for action in feasible_actions:
				agent_next_time = deepcopy(agent)

				# If the action involves picking up orders, update the agent's state accordingly
				if len(action[0]) != 0:
					agent_next_time.orders_to_pickup = deepcopy(action[0])

				# Simulate the agent's motion to get the next state after taking the action
				self.envt.simulate_vehicle_motion(agent_next_time, experience.time)

				# Add the post-action state of the agent to the list
				agents_post_actions.append(agent_next_time)

			# Ensure the number of post-action states matches the number of feasible actions
			assert len(feasible_actions) == len(agents_post_actions)

			# Append the list of post-action states for the current agent to the overall list
			all_agents_post_actions.append(agents_post_actions)

		# Calculate the time for the next state
		next_time = experience.time + self.envt.epoch_length

		return self._format_input_batch(all_agents_post_actions, next_time, experience.num_requests, experience.num_agents_on_break, experience.num_agents_at_warehouse, experience.avg_capacity_vehicles)

	def _format_input_batch(self, all_agents_post_actions, time, num_requests, num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles):
		inputs = []
		for next_agents in all_agents_post_actions:
			for agent in next_agents:
				# Normalizing and formatting agent's locations and delays
				locations_input = self._add_locations(agent.orders_to_pickup)
				delays_input = self._add_delays(agent.orders_to_pickup, time)

				# Normalizing current time, number of requests, and agents' statuses
				time_input = (time - self.envt.start_epoch) / (self.envt.stop_epoch - self.envt.start_epoch)
				num_requests_input = num_requests / 200
				num_agents_on_break_input = num_agents_on_break / self.envt.num_agents
				num_at_warehouse_input = num_agents_at_warehouse / self.envt.num_agents

				# Normalizing agents' shift time and time until return
				shift_time_input = agent.shift_start / 1440
				return_time_input = agent.time_until_return / 1440

				# Normalizing agents' current capacity and average capacity of vehicles
				capacity_input = len(agent.orders_to_pickup) / self.envt.car_capacity
				avg_capacity_vehicles_input = avg_capacity_vehicles / self.envt.car_capacity

				inputs.append([locations_input, delays_input, time_input, num_requests_input, num_agents_on_break_input, num_at_warehouse_input, shift_time_input, return_time_input, capacity_input, avg_capacity_vehicles_input])

		return inputs

	def _get_current_state_input(self, experience):
		current_agents = [[deepcopy(agent) for agent in experience.agents]]
		return self._format_input_batch(current_agents, experience.time, experience.num_past_requests, experience.num_past_on_break, experience.num_past_at_warehouse, experience.num_past_cap)


	def _add_locations(self, locations):
		# Initialize an array to hold location inputs with zeros
		locations_input = np.zeros(shape=(self.envt.car_capacity,), dtype='int32')

		# Iterate over each location in the provided list
		for loc_id in range(len(locations)):
			# Assign the location (destination + 1 to account for zero-indexing) to the array
			locations_input[loc_id] = locations[loc_id].destination + 1

		return locations_input

	def _add_delays(self, orders, current_time):
		# Initialize an array to hold delay inputs with -1 (indicating no delay/order)
		delays = np.zeros(shape=(self.envt.car_capacity, 1)) - 1

		# Iterate over each order
		for i, order in enumerate(orders):
			# Calculate the normalized delay for the order
			delays[i] = (order.deadline - current_time) / (order.deadline - order.origin_time)

		return delays

	def _process_experience_into_tensors(self, formatted_experience):
		# Initialize lists to hold each input tensor
		locations_input_list = []
		delays_input_list = []
		current_time_input_list = []
		num_requests_input_list = []
		num_on_break_input_list = []
		num_at_warehouse_input_list = []
		shift_time_input_list = []
		return_time_input_list = []
		capacity_input_list = []
		avg_capacity_vehicles_input_list = []

		for sample in formatted_experience:
			locations_input_list.append(torch.tensor(sample[0], dtype=torch.long))
			delays_input_list.append(torch.tensor(sample[1], dtype=torch.float32))
			current_time_input_list.append(torch.tensor([sample[2]], dtype=torch.float32))
			num_requests_input_list.append(torch.tensor([sample[3]], dtype=torch.float32))
			num_on_break_input_list.append(torch.tensor([sample[4]], dtype=torch.float32))
			num_at_warehouse_input_list.append(torch.tensor([sample[5]], dtype=torch.float32))
			shift_time_input_list.append(torch.tensor([sample[6]], dtype=torch.float32))
			return_time_input_list.append(torch.tensor([sample[7]], dtype=torch.float32))
			capacity_input_list.append(torch.tensor([sample[8]], dtype=torch.float32))
			avg_capacity_vehicles_input_list.append(torch.tensor([sample[9]], dtype=torch.float32))

		# Stack the lists into tensors
		locations_input = torch.stack(locations_input_list)  # Shape: (batch_size, car_capacity)
		delays_input = torch.stack(delays_input_list)  # Shape: (batch_size, car_capacity, 1)
		current_time_input = torch.stack(current_time_input_list)  # Shape: (batch_size, 1)
		num_requests_input = torch.stack(num_requests_input_list)  # Shape: (batch_size, 1)
		num_on_break_input = torch.stack(num_on_break_input_list)  # Shape: (batch_size, 1)
		num_at_warehouse_input = torch.stack(num_at_warehouse_input_list)  # Shape: (batch_size, 1)
		shift_time_input = torch.stack(shift_time_input_list)  # Shape: (batch_size, 1)
		return_time_input = torch.stack(return_time_input_list)  # Shape: (batch_size, 1)
		capacity_input = torch.stack(capacity_input_list)  # Shape: (batch_size, 1)
		avg_capacity_vehicles_input = torch.stack(avg_capacity_vehicles_input_list)  # Shape: (batch_size, 1)

		return (locations_input, delays_input, current_time_input,
				num_requests_input, num_on_break_input, num_at_warehouse_input,
				shift_time_input, return_time_input, capacity_input, avg_capacity_vehicles_input)


	def _reshape_values(self, values, shape):
		recon_values = []
		for agent_id in range(shape[0]):
			sz = shape[1][agent_id]
			recon_values.append(values[0:sz])
			values = values[sz:]
		return recon_values

	def _get_direct_rewards(self, agent_actions):
		rewards = []
		for actions in agent_actions:
			for act in actions:
				# rewards.append(len(act[1]) * self.envt.M - act[2])
				# rewards.append(len(act[1]) - (act[2] / 100))
				rewards.append(len(act[1]))
		return np.array(rewards)

	def pair_scores(self, scored_actions_all_agents, agents_matchings):
		final_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}
		id_to_pairings = {agent_id: {} for agent_id in range(len(agents_matchings))}

		for agent_id, (action_matchings, matching_scores) in enumerate(zip(agents_matchings, scored_actions_all_agents)):
			for action, score in zip(action_matchings, matching_scores):
				action_name = f"o_{'_'.join(map(str, action[1]))}"
				final_pairings[agent_id][action_name] = score
				id_to_pairings[agent_id][action_name] = action

		return final_pairings, id_to_pairings

	def remember(self, experience):
		self.replay_buffer.add(experience)

	def update(self):
		if self.num_min_train_samples > len(self.replay_buffer):
			return

		beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
		experiences, weights, batch_idxes = self.replay_buffer.sample(self.num_samples, beta)
		weights = torch.tensor(weights, dtype=torch.float32)

		total_loss = 0.0

		for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
			experience = deepcopy(experience)
			w = np.array([weights[experience_idx]] * self.envt.num_agents)

			scored_actions_all_agents = self.get_value(experience, use_target = True)
			final_pairings, id_to_pairings = self.pair_scores(scored_actions_all_agents, experience.feasible_actions_all_agents)
			matchings, scores = self.central_agent.choose_actions(final_pairings, id_to_pairings, len(experience.agents), experience.current_order_ids, is_training=False)
			scores = [scores[i] for i in range(len(scores))]
			supervised_targets = torch.tensor(scores, dtype=torch.float32).reshape(-1, 1)

			current_inputs = self._format_experience(experience, current_time = True)
			inputs = [x.to(self.device) for x in self._process_experience_into_tensors(current_inputs)]
			outputs = self.model(*inputs)

			loss_function = torch.nn.MSELoss(reduction='none')
			loss = loss_function(outputs, supervised_targets)
			weighted_loss = (loss * weights[experience_idx]).mean()

			total_loss += float(weighted_loss)

			# Backward pass and optimization
			self.optimizer.zero_grad()
			weighted_loss.backward()
			self.optimizer.step()

			# Update replay buffer weights
			predicted_values = self.model(*inputs)
			loss = float(torch.mean((predicted_values - supervised_targets) ** 2 + 1e-6))
			self.replay_buffer.update_priorities([batch_idx], [loss])

			self.soft_update()

		avg_loss = total_loss / self.num_samples
		self.scheduler.step(avg_loss)

		return avg_loss














