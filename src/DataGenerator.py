from RequestOrder import RequestOrder
import numpy

class DataGenerator(object):
    """
    A class to generate simulated data for requests (like transportation or delivery requests)
    based on various parameters.

    Attributes:
        data (str): City name of data used.
        variation_percentage (float): Percentage variation in data generation.
        speed_var (float): Speed variation factor.
        num_locations (int): Number of locations to consider.
        road_speed (float): Average road speed.
        epoch_length (int): Length of each time epoch in minutes.
        dist (list): Distribution of data across time.
        df (pandas.DataFrame): DataFrame containing location information.
        seed (int): Seed for random number generation.
        np (numpy.random.RandomState): Numpy random state for generating random numbers.
    """

    def __init__(self, data, variation_percentage, speed_var, num_locations, road_speed, epoch_length, dist, df, seed):
        """
        The constructor for DataGenerator class.

        Parameters:
            data (str): City name of data used.
            variation_percentage (float): Percentage variation in data generation.
            speed_var (float): Speed variation factor.
            num_locations (int): Number of locations to consider.
            road_speed (float): Average road speed.
            epoch_length (int): Length of each time epoch in minutes.
            dist (list): Distribution of data across time.
            df (pandas.DataFrame): DataFrame containing location information.
            seed (int): Seed for random number generation.
        """
        self.data = data
        self.variation_percentage = variation_percentage
        self.speed_var = speed_var
        self.num_locations = num_locations
        self.road_speed = road_speed
        self.epoch_length = epoch_length
        self.dist = dist
        self.df = df
        self.seed = seed
        self.np = numpy.random.RandomState(seed)

    def get_requests(self, time):
        """
        Generates a list of request orders for a specific time.

        Parameters:
            time (int): The specific time for which to generate requests.

        Returns:
            list[RequestOrder]: A list of generated request orders.
        """
        number_of_requests = self.get_number_orders(time) + 1
        locations = self.get_locations(number_of_requests)
        return self.create_requests(time, locations)

    def get_number_orders(self, time, std=1):
        """
        Calculates the number of orders for a given time based on a normal distribution.

        Parameters:
            time (int): The specific time for which to calculate the number of orders.
            std (int, optional): Standard deviation for the normal distribution. Defaults to 1.

        Returns:
            int: The calculated number of orders.
        """
        avg = self.dist[time]
        values = [v for v in range(avg - std, avg + std + 1) if v >= 0]
        return self.normal_distributed_random_element(values)

    def normal_distributed_random_element(self, x):
        """
        Generates a normally distributed random element from a given list.

        Parameters:
            x (list[int]): The list of values from which to select.

        Returns:
            int: A randomly selected element from the list.
        """
        mean = numpy.mean(range(len(x)))
        std = numpy.std(range(len(x)))
        index = int(self.np.normal(mean, std))
        while index >= len(x):
            index = int(self.np.normal(mean, std))
        return x[int(index)]
    
    def get_locations(self, num_requests):
        """
        Randomly selects locations for a specified number of requests.

        Parameters:
            num_requests (int): The number of requests for which to select locations.

        Returns:
            numpy.ndarray: An array of selected locations.
        """
        return self.np.choice(self.df.index, size=num_requests, p=self.df.prevalence)

    def create_requests(self, time, locations):
        """
        Creates a list of RequestOrder objects based on given time and locations.

        Parameters:
            time (int): The specific time for the requests.
            locations (list): The list of locations for the requests.

        Returns:
            list[RequestOrder]: A list of RequestOrder objects.
        """
        return [RequestOrder(time, loc) for loc in locations]

    def get_shift_start_time(self, num_agents=10, sched_type='poor', mean=0, std_dev=8):
        """
        Determines the start times of shifts based on the number of agents and scheduling type.

        Parameters:
            num_agents (int, optional): The number of agents. Defaults to 10.
            sched_type (str, optional): The type of scheduling ('realistic', 'uniform', 'poor'). Defaults to 'poor'.
            mean (int, optional): The mean for shift start time calculation. Defaults to 0.
            std_dev (int, optional): The standard deviation for shift start time calculation. Defaults to 8.

        Returns:
            list[int]: A list of start times for the agents.
        """
        # Different scheduling types and the number of agents affect the start times.
        # Implementation details for each scheduling type and number of agents.
        if num_agents == 10:
            if sched_type == 'realistic':
                starts = [0, 0, 6, 10, 12, 16, 17, 17, 18, 18]
            elif sched_type == 'uniform':
                starts = [0, 0, 6, 6, 10, 12, 12, 16, 18, 18]
            elif sched_type == 'poor':
                s = [0, 0, 6, 10, 12, 16, 17, 17, 18, 18]
                starts = [abs(i - 18) for i in s]
        elif num_agents == 15:
            if sched_type == 'realistic':
                starts = [0, 0, 6, 10, 10, 12, 12, 16, 16, 17, 17, 17, 17, 18, 18]
            elif sched_type == 'uniform':
                starts = [0, 0, 0, 0, 6, 6, 6, 10, 12, 12, 12, 16, 18, 18, 18]
            elif sched_type == 'poor':
                s = [0, 0, 6, 10, 10, 12, 12, 16, 16, 17, 17, 17, 17, 18, 18]
                starts = [abs(i - 18) for i in s]
        elif num_agents == 20:
            if sched_type == 'realistic':
                starts = [0, 0, 0, 6, 10, 10, 12, 12, 12, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18]
            elif sched_type == 'uniform':
                starts = [0, 0, 0, 0, 0, 6, 6, 6, 6, 10, 12, 12, 12, 12, 16, 16, 18, 18, 18, 18]
            elif sched_type == 'poor':
                s = [0, 0, 0, 6, 10, 10, 12, 12, 12, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18]
                starts = [abs(i - 18) for i in s]

        assert len(starts) == num_agents
        return starts

    def create_test_scenarios(self, num_days, epoch_length):
        """
        Creates test scenarios for a given number of days.

        Parameters:
            num_days (int): The number of days for which to create test scenarios.
            epoch_length (int): The length of each epoch in minutes.

        Returns:
            dict: A dictionary of test scenarios, keyed by day and time.
        """
        test_scenarios = {day: {} for day in range(num_days)}
        for day in range(num_days):
            for time in range(0, 1440, epoch_length):
                test_scenarios[day][time] = self.get_requests(time)
        return test_scenarios
