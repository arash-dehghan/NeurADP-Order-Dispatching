import sys
sys.dont_write_bytecode = True
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import pickle
from Environment import Environment
from CentralAgent import CentralAgent
from ValueFunction import NeurADP
from LearningAgent import LearningAgent
from Experience import Experience
from ResultCollector import ResultCollector
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def run_epoch(envt, central_agent, value_function, requests, request_generator, agents_predefined, is_training = False):
	"""
    Runs a single epoch of the simulation.

    Parameters:
        envt (Environment): The simulation environment.
        central_agent (CentralAgent): The central agent managing order assignments.
        value_function (NeurADP): The neural network-based value function.
        requests (list): List of requests if not training; otherwise, None.
        request_generator (RequestGenerator): Generator for creating requests during training.
        agents_predefined (list[LearningAgent]): Predefined agents for the simulation.
        is_training (bool, optional): Whether the epoch is for training. Defaults to False.

    Returns:
        tuple: A tuple containing arrays of the results and statistics from the epoch.
    """
	envt.current_time = envt.start_epoch
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	Experience.envt = envt
	agents = deepcopy(agents_predefined)
	global_order_id, total_orders_served = 0, 0
	graph_seen, graph_served, times = ([] for _ in range(3))
	num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap = 0, 0, 0, 0
	nums_at_warehouse, nums_on_break, nums_delivering, avg_times_to_return, avg_queue_size, avg_queue_travel_time, avg_matching_size, orders_served_timings = ([] for _ in range(8))

	for t in range(ts):
		# Generate and add deadlines to new orders, add new orders to remaining orders
		if is_training:
			current_orders = [central_agent.set_deadlines(order, i) for i, order in enumerate(request_generator.get_requests(envt.current_time), start=global_order_id)]
		else:
			current_orders = [central_agent.set_deadlines(order, i) for i,order in enumerate(requests[envt.current_time], start=global_order_id)]
		global_order_id += len(current_orders)
		current_order_ids = [order.id for order in current_orders]

		# Get some statistics
		if not is_training:
			nums_at_warehouse.append(sum([1 for agent in agents if central_agent._check_at_warehouse_status(agent)]))
			nums_on_break.append(sum([1 for agent in agents if central_agent._check_break_status(agent)]))
			nums_delivering.append(len(agents) - nums_at_warehouse[-1] - nums_on_break[-1])
			return_times = [agent.time_until_return for agent in agents if (not central_agent._check_break_status(agent)) and (not central_agent._check_at_warehouse_status(agent))]
			avg_times_to_return.append(np.mean(return_times) if len(return_times) > 0 else 0)
			queue_sizes = [len(agent.orders_to_pickup) for agent in agents if len(agent.orders_to_pickup) > 0]
			avg_queue_size.append(np.mean(queue_sizes) if len(queue_sizes) > 0 else 0)
			queue_times = [envt._get_ordering_return_time(agent.orders_to_pickup) for agent in agents if len(agent.orders_to_pickup) > 0]
			avg_queue_travel_time.append(np.mean(queue_times) if len(queue_times) > 0 else 0)

		# Get other external info to add to post-decision state
		num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles = central_agent.get_external_infor(agents)

		# Get feasible actions for each agent
		feasible_actions = central_agent.get_feasible_actions(agents, current_orders)

		# Create Experience
		experience = Experience(deepcopy(agents), feasible_actions, envt.current_time, len(current_orders), current_order_ids, num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles, num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap)

		# Score the feasible actions and pair taking action with its score
		scored_actions_all_agents = value_function.get_value([experience])
		final_pairings, id_to_pairings = value_function.pair_scores(scored_actions_all_agents,feasible_actions)

		# Choose actions for each agent
		matchings, scores = central_agent.choose_actions(final_pairings, id_to_pairings, len(agents), current_order_ids, is_training=is_training)

		# Get some further statistics
		if not is_training:
			orders_timings = [(o.deadline - o.origin_time) for agent_id, match in matchings.items() for o in match[0]]
			orders_served_timings.append(np.mean(orders_timings) if len(orders_timings) > 0 else 0)
			servings = [len(match[1]) for agent_id,match in matchings.items() if len(match[1]) > 0]
			avg_matching_size.append(np.mean(servings) if len(servings) > 0 else 0)

		# Update if training
		if is_training:
			if t > 0:
				# Update replay buffer
				value_function.remember(experience)
			# Update value function every TRAINING_FREQUENCY timesteps
			if ((int(envt.current_time) / int(envt.epoch_length)) % 1 == 0):
				value_function.update(central_agent)

		# Set the new trajectories for each agent
		orders_served = central_agent.set_new_paths(agents, matchings)
		total_orders_served += orders_served
		graph_seen.append(len(current_orders))
		graph_served.append(orders_served)
		times.append(envt.current_time)

		# Update the time
		envt.current_time += envt.epoch_length
		num_past_requests = len(current_orders)
		num_past_at_warehouse = num_agents_at_warehouse
		num_past_on_break = num_agents_on_break
		num_past_cap = avg_capacity_vehicles

	if is_training:
		envt.num_days_trained += 1
	
	return np.array([graph_served, graph_seen]), np.array([nums_at_warehouse, nums_on_break, nums_delivering, avg_times_to_return, avg_queue_size, avg_queue_travel_time, avg_matching_size, orders_served_timings])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=15)
	parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
	parser.add_argument('-shift_length', '--shift_length', type=int , default=6)
	parser.add_argument('-variation_percentage', '--variation_percentage', type=float , default=0.2)
	parser.add_argument('-speed_var', '--speed_var', type=float , default=0.3)
	parser.add_argument('-num_locations', '--num_locations', type=int , default=1000)
	parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-dt', '--delaytime', type=float, default=10)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=3)
	parser.add_argument('-shift_style', '--shift_style', type=str, default='realistic')
	parser.add_argument('-train_days', '--train_days', type=int, default=60)
	parser.add_argument('-test_days', '--test_days', type=int, default=20)
	parser.add_argument('-test_every', '--test_every', type=int, default=5)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	parser.add_argument('-trainable', '--trainable', type=int, default=1)
	args = parser.parse_args()
	assert args.numagents in [10,15,20]
	assert args.shift_length == 6
	assert args.shift_style in ['realistic', 'uniform', 'poor']

	filename = f"{args.data}_{args.epoch_length}_{args.road_speed}_{args.num_locations}_{args.seed}"
	request_generator = pickle.load(open(f'../data/generations/{filename}/data_{filename}.pickle','rb'))
	envt = Environment(args.numagents, args.epoch_length, args.capacity, args.data, args.shift_length, args.road_speed)
	central_agent = CentralAgent(envt, args.numagents, args.delaytime, args.capacity)
	value_function = NeurADP(envt, args.delaytime, args.trainable, filename)
	test_data = request_generator.create_test_scenarios(args.test_days, args.epoch_length)

	test_shift_start_times = request_generator.get_shift_start_time(num_agents = args.numagents, sched_type=args.shift_style)
	stops = [i for i in range(args.test_every,args.train_days + args.test_every,args.test_every)]
	result_collector = ResultCollector()
	final_results = []
	i = 0
	tots = []
	fs = []
	test_results = {day: [] for day in range(args.test_days)}
	test_days_seen = {}
	results_nums_at_warehouse = {day: {} for day in range(args.test_days)}
	results_nums_on_break = {day: {} for day in range(args.test_days)}
	results_nums_delivering = {day: {} for day in range(args.test_days)}
	results_avg_times_to_return = {day: {} for day in range(args.test_days)}
	results_avg_queue_size = {day: {} for day in range(args.test_days)}
	results_avg_queue_travel_time = {day: {} for day in range(args.test_days)}
	results_avg_matching_size = {day: {} for day in range(args.test_days)}
	results_orders_served_timings = {day: {} for day in range(args.test_days)}

	# Initial myopic results
	for test_day in tqdm(range(args.test_days)):
		orders = test_data[test_day]
		agents = [LearningAgent(agent_id, args.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
		results, stats = run_epoch(envt, central_agent, value_function, orders, request_generator, agents, False)
		fs.append(sum(results[0]))
		test_results[test_day].append(sum(results[0]))
		test_days_seen[test_day] = sum(results[1])
		final_results.append(results)
		results_nums_at_warehouse[test_day][i] = stats[0]
		results_nums_on_break[test_day][i] = stats[1]
		results_nums_delivering[test_day][i] = stats[2]
		results_avg_times_to_return[test_day][i] = stats[3]
		results_avg_queue_size[test_day][i] = stats[4]
		results_avg_queue_travel_time[test_day][i] = stats[5]
		results_avg_matching_size[test_day][i] = stats[6]
		results_orders_served_timings[test_day][i] = stats[7]
	tots.append(np.mean(fs))
	result_collector.test_days_seen = test_days_seen
	result_collector.update_results(i, final_results)
	print(tots)

	# Train the model
	for train_day in tqdm(range(args.train_days)):
		agents = [LearningAgent(agent_id, args.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
		run_epoch(envt, central_agent, value_function, None, request_generator, agents, True)
		i += 1
		if i in stops:
			final_results = []
			fs = []
			# Get the test results
			for test_day in range(args.test_days):
				orders = test_data[test_day]
				agents = [LearningAgent(agent_id, args.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
				results, stats = run_epoch(envt, central_agent, value_function, orders, request_generator, agents, False)
				fs.append(sum(results[0]))
				test_results[test_day].append(sum(results[0]))
				final_results.append(results)
				results_nums_at_warehouse[test_day][i] = stats[0]
				results_nums_on_break[test_day][i] = stats[1]
				results_nums_delivering[test_day][i] = stats[2]
				results_avg_times_to_return[test_day][i] = stats[3]
				results_avg_queue_size[test_day][i] = stats[4]
				results_avg_queue_travel_time[test_day][i] = stats[5]
				results_avg_matching_size[test_day][i] = stats[6]
				results_orders_served_timings[test_day][i] = stats[7]
			tots.append(np.mean(fs))
			result_collector.update_results(i, final_results)
			print(tots)

	file_data = f'{args.data}_{args.numagents}_{value_function.M}_{int(args.road_speed)}_{args.shift_length}_{args.delaytime}_{args.epoch_length}_{args.capacity}_{args.seed}_{args.trainable}_{args.shift_style}'
	result_collector.test_dbd = test_results
	result_collector.results_nums_at_warehouse = results_nums_at_warehouse
	result_collector.results_nums_on_break = results_nums_on_break
	result_collector.results_nums_delivering = results_nums_delivering
	result_collector.results_avg_times_to_return = results_avg_times_to_return
	result_collector.results_avg_queue_size = results_avg_queue_size
	result_collector.results_avg_queue_travel_time = results_avg_queue_travel_time
	result_collector.results_avg_matching_size = results_avg_matching_size
	result_collector.results_avg_orders_served_timings = results_orders_served_timings
	value_function.model.save(f'../Results/NeurADP_{file_data}.h5')
	with open(f'../Results/NeurADP_{file_data}.pickle', 'wb') as handle:
		pickle.dump(result_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)

