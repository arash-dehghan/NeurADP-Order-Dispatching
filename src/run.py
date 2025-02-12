import sys
sys.dont_write_bytecode = True
import argparse
import pickle
from Environment import Environment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Experience import Experience
from NeurADP import NeurADP
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch

def run_test(test_data, test_shift_start_times):
	test_data, test_shift_start_times = deepcopy(test_data), deepcopy(test_shift_start_times)
	results_served, results_seen = [], []
	for test_day in tqdm(range(args.test_days)):
		orders = test_data[test_day]
		agents = [LearningAgent(agent_id, envt.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
		results = run_epoch(envt, central_agent, value_function, orders, request_generator, agents, False)
		results_served.append(results[0])
		results_seen.append(results[1])
	return (np.mean(results_served), np.mean(results_seen))


def run_epoch(envt, central_agent, value_function, requests, request_generator, agents_predefined, is_training = False):
	envt.current_time = envt.start_epoch
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	Experience.envt = envt
	agents = deepcopy(agents_predefined)
	global_order_id, total_orders_served = 0, 0
	num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap = 0, 0, 0, 0
	graph_seen, graph_served, times = ([] for _ in range(3))

	for t in range(ts):
		# Get orders
		current_orders = [central_agent.set_deadlines(order, i) for i, order in enumerate(request_generator.get_requests(envt.current_time), start=global_order_id)] if is_training else [central_agent.set_deadlines(order, i) for i,order in enumerate(requests[envt.current_time], start=global_order_id)]
		global_order_id += len(current_orders)
		current_order_ids = [order.id for order in current_orders]

		# Get other external info to add to post-decision state
		num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles = central_agent.get_external_infor(agents)

		# Get feasible actions for each agent
		feasible_actions = central_agent.get_feasible_actions(agents, current_orders)

		# Create Experience
		experience = Experience(deepcopy(agents), deepcopy(feasible_actions), deepcopy(current_order_ids), deepcopy(envt.current_time), len(current_orders), deepcopy(num_agents_on_break), deepcopy(num_agents_at_warehouse), deepcopy(avg_capacity_vehicles), deepcopy(num_past_requests), deepcopy(num_past_at_warehouse), deepcopy(num_past_on_break), deepcopy(num_past_cap))

		# Score the feasible actions and pair taking action with its score
		scored_actions_all_agents = value_function.get_value(experience)
		final_pairings, id_to_pairings = value_function.pair_scores(scored_actions_all_agents, feasible_actions)

		# Choose actions for each agent
		matchings, scores = central_agent.choose_actions(final_pairings, id_to_pairings, len(agents), current_order_ids, is_training=is_training)

		# Update if training
		if is_training:
			if t > 0:
				value_function.remember(experience)
			value_function.update()

		# Set the new trajectories for each agent
		orders_served, _ , _ = central_agent.set_new_paths(agents, matchings, envt.current_time)
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


	return (sum(graph_served), sum(graph_seen))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=15)
	parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
	parser.add_argument('-num_locations', '--num_locations', type=int , default=1000)
	parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
	parser.add_argument('-boundedness', '--boundedness', type=str, default='unbounded')
	parser.add_argument('-delay_type', '--delay_type', type=str, choices=['ultra', 'normal'], default='normal')
	parser.add_argument('-breaks_included', '--breaks_included', type=int, default=0)
	parser.add_argument('-dt', '--delaytime', type=float, default=10)
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=3)
	parser.add_argument('-shift_style', '--shift_style', type=str, default='realistic')
	parser.add_argument('-train_days', '--train_days', type=int, default=300)
	parser.add_argument('-test_days', '--test_days', type=int, default=10)
	parser.add_argument('-test_every', '--test_every', type=int, default=5)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	args = parser.parse_args()
	assert args.numagents in [10,15,20]
	assert args.shift_style in ['realistic', 'uniform', 'poor']

	# Load generator based on file name
	filename = f'{args.data}_{args.epoch_length}_{args.road_speed}_{args.num_locations}_{args.boundedness}_{args.delaytime}_{args.seed}'
	file_data = f'{filename}_{args.delay_type}_{args.capacity}_{args.numagents}_{args.breaks_included}'
	request_generator = pickle.load(open(f'../data/generations/{filename}/data_{filename}.pickle','rb'))

	# Define necessary data objects
	envt = Environment(filename, args.numagents, args.epoch_length, args.capacity, args.data, args.road_speed, args.delay_type, args.breaks_included)
	central_agent = CentralAgent(envt)
	value_function = NeurADP(envt, central_agent)

	# torch.save(value_function.model.state_dict(), f'../Results/Myopic_weights_{file_data}.pt')
	# value_function.model.load_state_dict(torch.load(f'../Results/NeurADP_weights_{file_data}.pt'))
	# value_function.target.load_state_dict(torch.load(f'../Results/NeurADP_weights_{file_data}.pt'))

	# Generate testing orders and shifts
	test_data = request_generator.create_test_scenarios(args.test_days, args.epoch_length)
	test_shift_start_times = request_generator.get_shift_start_time(num_agents = args.numagents, sched_type=args.shift_style)

	# Define information for training iterations
	stops = range(args.test_every, args.train_days + 1, args.test_every)
	final_results, best_served = [], 0

	# Run initial test
	results = run_test(test_data, test_shift_start_times)
	final_results.append(results)
	print(final_results)
	# exit()

	# Run training iterations
	for train_day, _ in enumerate(tqdm(range(args.train_days)), start=1):
		agents = [LearningAgent(agent_id, envt.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
		_ = run_epoch(envt, central_agent, value_function, None, request_generator, agents, True)
		if train_day in stops:
			results = run_test(test_data, test_shift_start_times)
			final_results.append(results)
			print(final_results)
			if results[0] > best_served:
				best_served = results[0]
				torch.save(value_function.model.state_dict(), f'../Results/NeurADP_weights_{file_data}.pt')











