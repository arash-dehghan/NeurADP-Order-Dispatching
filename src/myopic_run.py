import sys
sys.dont_write_bytecode = True
import argparse
import pickle
from Environment import Environment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Experience import Experience
from copy import deepcopy
from tqdm import tqdm
import numpy as np

def run_test(test_data, test_shift_start_times, sort_type, variation):
	test_data, test_shift_start_times = deepcopy(test_data), deepcopy(test_shift_start_times)
	results_served, results_seen = [], []
	for test_day in tqdm(range(args.test_days)):
		orders = test_data[test_day]
		agents = [LearningAgent(agent_id, envt.shift_length, test_shift_start_times[agent_id]) for agent_id in range(args.numagents)]
		results = run_epoch(envt, central_agent, orders, agents, sort_type, variation)
		results_served.append(results[0])
		results_seen.append(results[1])
	return (np.mean(results_served), np.mean(results_seen), np.std(results_served), np.std(results_seen))

def sort_agents(agents, time, sort_type, variation):
	if sort_type == 'capacity':
		sorted_agents = sorted(agents, key=lambda agent: len(agent.orders_to_pickup)) if variation == 'emptiest' else sorted(agents, key=lambda agent: len(agent.orders_to_pickup), reverse=True)
	elif sort_type == 'proximity':
		set_proximities(agents, time)
		sorted_agents = sorted(agents, key=lambda agent: agent.proximity) if variation == 'closest' else sorted(agents, key=lambda agent: agent.proximity, reverse=True)
	return sorted_agents

def set_proximities(agents, time):
	for agent in agents:
		if (time < agent.shift_start) or (time >= agent.shift_end):
			agent.proximity = (agent.shift_start - time) if (time < agent.shift_start) else 1000
		else:
			agent.proximity = agent.time_until_return

def filter_actions(actions, orders_fulfilled):
	filtered_actions = []
	for action in actions:
		if all(value not in orders_fulfilled for value in action[1]):
			filtered_actions.append(action)
	return filtered_actions

def get_best_action(actions):
	actions = sorted(actions, key=lambda x: (-len(x[1]), x[2]))
	# actions = sorted(actions, key=lambda x: (-len(x[1])))
	return actions[0][0], actions[0][1]

def match_agents(Lagents, actions, time, sort_type, variation):
	actions = {agent_id : action_list for agent_id, action_list in enumerate(actions)}
	sorted_agents = sort_agents(deepcopy(Lagents), time, sort_type, variation)
	orders_fulfilled = []
	agent_action_matchings = {}
	for agent in sorted_agents:
		agent_actions = actions[agent.id]
		filtered_agent_actions = filter_actions(agent_actions, orders_fulfilled)
		act, orders_served = get_best_action(filtered_agent_actions)
		orders_fulfilled += orders_served
		agent_action_matchings[agent.id] = (act, orders_served)

	assert len(orders_fulfilled) == len(set(orders_fulfilled))

	return agent_action_matchings, len(orders_fulfilled)

def run_epoch(envt, central_agent, requests, agents_predefined, sort_type, variation):
	envt.current_time = envt.start_epoch
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	Experience.envt = envt
	agents = deepcopy(agents_predefined)
	global_order_id, total_orders_served = 0, 0
	num_past_requests, num_past_at_warehouse, num_past_on_break, num_past_cap = 0, 0, 0, 0
	graph_seen, graph_served, times = ([] for _ in range(3))

	for t in range(ts):
		# Get orders
		current_orders = [central_agent.set_deadlines(order, i) for i,order in enumerate(requests[envt.current_time], start=global_order_id)]
		global_order_id += len(current_orders)
		current_order_ids = [order.id for order in current_orders]

		# Get feasible actions for each agent
		feasible_actions = central_agent.get_feasible_actions(agents, current_orders)

		matchings, orders_served = match_agents(agents, feasible_actions, envt.current_time, sort_type, variation)

		assert orders_served <= len(current_orders)

		# Set the new trajectories for each agent
		orders_served = central_agent.set_new_paths(agents, matchings, envt.current_time)
		total_orders_served += orders_served
		graph_seen.append(len(current_orders))
		graph_served.append(orders_served)
		times.append(envt.current_time)

		# Update the time
		envt.current_time += envt.epoch_length

	return (sum(graph_served), sum(graph_seen))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=15)
	parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
	parser.add_argument('-num_locations', '--num_locations', type=int , default=1000)
	parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
	parser.add_argument('-boundedness', '--boundedness', type=str, default='unbounded')
	parser.add_argument('-delay_type', '--delay_type', type=str, choices=['getir', 'normal'], default='normal')
	parser.add_argument('-breaks_included', '--breaks_included', type=int, default=1)
	parser.add_argument('-dt', '--delaytime', type=float, default=10)
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=4)
	parser.add_argument('-shift_style', '--shift_style', type=str, default='realistic')
	parser.add_argument('-train_days', '--train_days', type=int, default=200)
	parser.add_argument('-test_days', '--test_days', type=int, default=20)
	parser.add_argument('-test_every', '--test_every', type=int, default=5)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	args = parser.parse_args()
	assert args.numagents in [10,15,20]
	assert args.shift_style in ['realistic', 'uniform', 'poor']

	sort_type = 'proximity' # (capacity, proximity)
	variation = 'closest' # [(emptiest, fullest),(closest, farthest)]

	# Load generator based on file name
	filename = f'{args.data}_{args.epoch_length}_{args.road_speed}_{args.num_locations}_{args.boundedness}_{args.delaytime}_{args.seed}'
	file_data = f'{filename}_{args.delay_type}_{args.capacity}_{args.numagents}_{args.breaks_included}'
	request_generator = pickle.load(open(f'../data/generations/{filename}/data_{filename}.pickle','rb'))

	# Define necessary data objects
	envt = Environment(filename, args.numagents, args.epoch_length, args.capacity, args.data, args.road_speed, args.delay_type, args.breaks_included)
	central_agent = CentralAgent(envt)

	# Generate testing orders and shifts
	test_data = request_generator.create_test_scenarios(args.test_days, args.epoch_length)
	test_shift_start_times = request_generator.get_shift_start_time(num_agents = args.numagents, sched_type=args.shift_style)

	# Define information for training iterations
	stops = range(args.test_every, args.train_days + 1, args.test_every)
	final_results, best_served = [], 0

	# Run initial test
	results = run_test(test_data, test_shift_start_times, sort_type, variation)
	print(f'Avg Seen: {round(results[1],2)} +/- {round(results[3],2)}, Avg Served: {round(results[0],2)} +/- {round(results[2],2)}')




