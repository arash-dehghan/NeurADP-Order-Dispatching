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
from DataExtractor import Extractor
from collections import Counter

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

def add_order_matched_per_courier(matchings):
	for agent_id, contents in matchings.items():
		orders_matched_per_courier[agent_id] += len(contents[1])

def add_matching_size(time, matchings):
	sizes = [len(m[1]) for m in matchings.values() if len(m[1]) > 0]
	average_matching_size_throught_day[time] += sizes

def add_locations(orders, matchings):
	global all_locations_served
	orders = {order.id : order.destination for order in orders}
	all_locations_served += [orders[m_id] for m in matchings.values() for m_id in m[1]]

def add_average_trip_durations(time, trip_durations):
	average_trip_durations[time] += trip_durations

def add_send_off_sizes(time, send_off_sizes):
	average_send_off_sizes[time] += send_off_sizes

def add_delay_deadline_utilized(time, delay_deadline_utilized):
	average_delay_deadline_utilized[time] += delay_deadline_utilized

def add_revenue(time, matchings):
	for agent_id, matches in matchings.items():
		for new_match_id in matches[1]:
			for order in matches[0]:
				if order.id == new_match_id:
					average_revenue_by_courier[agent_id] += order.revenue
					average_revenue_by_time[time] += order.revenue

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

		# Get other external info to add to post-decision state
		num_agents_on_break, num_agents_at_warehouse, avg_capacity_vehicles = central_agent.get_external_infor(agents)

		# Get feasible actions for each agent
		feasible_actions = central_agent.get_feasible_actions(agents, current_orders)

		matchings, orders_served = match_agents(agents, feasible_actions, envt.current_time, sort_type, variation)

		assert orders_served <= len(current_orders)

		# Set the new trajectories for each agent
		orders_served, trip_durations, send_off_sizes, delay_deadline_utilized = central_agent.set_new_paths(agents, matchings, envt.current_time)
		
		total_orders_served += orders_served
		graph_seen.append(len(current_orders))
		graph_served.append(orders_served)
		times.append(envt.current_time)

		##### STATS #####
		orders_served_throughout_day[envt.current_time] += (orders_served / args.test_days)
		orders_seen_throughout_day[envt.current_time] += (len(current_orders) / args.test_days)
		couriers_at_depot_throughout_day[envt.current_time] += (num_agents_at_warehouse / args.test_days)
		add_order_matched_per_courier(matchings)
		add_matching_size(envt.current_time, matchings)
		add_locations(current_orders, matchings)
		add_average_trip_durations(envt.current_time, trip_durations)
		add_send_off_sizes(envt.current_time, send_off_sizes)
		add_delay_deadline_utilized(envt.current_time, delay_deadline_utilized)
		add_revenue(envt.current_time, matchings)

		# Update the time
		envt.current_time += envt.epoch_length

	return (sum(graph_served), sum(graph_seen))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=15)
	parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
	parser.add_argument('-num_locations', '--num_locations', type=int , default=1000)
	parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
	parser.add_argument('-boundedness', '--boundedness', type=str, default='bounded')
	parser.add_argument('-delay_type', '--delay_type', type=str, choices=['getir', 'normal'], default='getir')
	parser.add_argument('-breaks_included', '--breaks_included', type=int, default=1)
	parser.add_argument('-dt', '--delaytime', type=float, default=10)
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=3)
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

	##### STATS #####
	orders_served_throughout_day = {time: 0 for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	orders_seen_throughout_day = {time: 0 for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	couriers_at_depot_throughout_day = {time: 0 for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}	
	orders_matched_per_courier = {courier_id : 0 for courier_id in range(args.numagents)}
	average_matching_size_throught_day = {time: [] for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	all_locations_served = []
	average_trip_durations = {time: [] for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	average_send_off_sizes = {time: [] for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	average_delay_deadline_utilized = {time: [] for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}
	average_revenue_by_courier = {i : 0 for i in range(args.numagents)}
	average_revenue_by_time = {time: 0 for time in range(0, int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length) * envt.epoch_length, envt.epoch_length)}

	# Run initial test
	results = run_test(test_data, test_shift_start_times, sort_type, variation)

	final_data = {'Average Seen' : round(results[1],2),
	'Average Seen STD' : round(results[3],2),
	'Average Served' : round(results[0],2),
	'Average Served STD' : round(results[2],2),
	'Orders Served Throughout Day' : orders_served_throughout_day,
	'Orders Seen Throughout Day' : orders_seen_throughout_day,
	'Couriers at Depot Throughout Day': couriers_at_depot_throughout_day,
	'Orders Matched Per Courier' : orders_matched_per_courier,
	'Average Matching Size Throughout Day' : average_matching_size_throught_day,
	'All Locations Served' : all_locations_served,
	'Average Trip Durations' : average_trip_durations,
	'Average Send Off Size' : average_send_off_sizes,
	'Average Delay Deadline Utilized' : average_delay_deadline_utilized,
	'Average Revenue by Courier' : average_revenue_by_courier,
	'Average Revenue by Time' : average_revenue_by_time}

	extractor = Extractor('Myopic', final_data)

	with open(f'../Statistics/myopic_{file_data}.pkl', 'wb') as file: pickle.dump(extractor, file)

	print(f'Avg Seen: {round(results[1],2)} +/- {round(results[3],2)}, Avg Served: {round(results[0],2)} +/- {round(results[2],2)}')




