import numpy as np

class ResultCollector(object):
    """
    A class for collecting and storing results from simulation iterations. 
    It tracks the number of requests seen and served each day and overall.

    Attributes:
        requests_seen_day_results (dict): Dictionary storing the number of requests seen each day.
        requests_served_day_results (dict): Dictionary storing the number of requests served each day.
        requests_seen (dict): Dictionary storing the total number of requests seen in each iteration.
        requests_served (dict): Dictionary storing the total number of requests served in each iteration.
    """

    def __init__(self):
        """
        Constructor for the ResultCollector class.
        Initializes dictionaries to store results.
        """
        self.requests_seen_day_results = {}  # Stores results for requests seen each day
        self.requests_served_day_results = {}  # Stores results for requests served each day
        self.requests_seen = {}  # Stores total requests seen per iteration
        self.requests_served = {}  # Stores total requests served per iteration

    def update_results(self, iteration, results):
        """
        Updates the result dictionaries with new data from a simulation iteration.

        Parameters:
            iteration (int): The iteration number of the simulation.
            results (list[tuple]): A list of tuples containing daily results for seen and handled requests.
        """
        # Calculate average seen and handled requests per day in the current iteration
        seen = np.sum([days_result[1] for days_result in results], 0) / len(results)
        handled = np.sum([days_result[0] for days_result in results], 0) / len(results)

        # Update the daily results for the current iteration
        self.requests_seen_day_results[iteration] = seen 
        self.requests_served_day_results[iteration] = handled

        # Update the total results for the current iteration
        self.requests_seen[iteration] = sum(np.sum([days_result[1] for days_result in results], 0))
        self.requests_served[iteration] = sum(np.sum([days_result[0] for days_result in results], 0))
