import sys
sys.dont_write_bytecode = True
import pandas as pd
import argparse
import numpy
import matplotlib.pyplot as plt
import pickle
from haversine import haversine
from DataGenerator import DataGenerator
import os

class DataPreparation(object):
    """
    A class to prepare data for the DataGenerator.

    Attributes:
        df (pandas.DataFrame): DataFrame with coordinates and prevalence of each location.
        dist (dict): Dictionary containing the distribution of orders across time.
    """

    def __init__(self, initial_vars):
        """
        Constructor for DataPreparation class.

        Parameters:
            initial_vars (dict): Dictionary of initial variables.
        """
        for key in initial_vars:
            setattr(self, key, initial_vars[key])
        self.df = self.create_dataframe()
        self.df.to_csv('location_coordinates.csv')
        self.filename = f'{self.data}_{self.epoch_length}_{self.road_speed}_{self.num_locations}_{"bounded" if self.bounded else "unbounded"}_{self.delay_bound}_{self.seed}'
        self.create_traveltime_file()
        self.dist = self.get_ultra_order_distribution()

    def get_variation(self, lats, lons):
        """
        Adds variation to the geographical location based on the latitude and longitude ranges.

        Parameters:
            lats (list[float]): List of latitudes.
            lons (list[float]): List of longitudes.

        Returns:
            tuple: A tuple with modified latitude and longitude.
        """
        lat_band, lon_band =  abs((max(lats) - min(lats)) * self.variation_percentage), abs((max(lons) - min(lons)) * self.variation_percentage)
        avg_lat, avg_lon = numpy.average(lats), numpy.average(lons)
        return (round(np.uniform(-lat_band, lat_band) + avg_lat, 4), round(np.uniform(-lon_band, lon_band) + avg_lon, 4))

    def get_centre_point(self, locations):
        """
        Calculates the centre point of all locations with added variation.

        Parameters:
            locations (list[tuple]): List of locations (latitude, longitude).

        Returns:
            list: A list containing the centre point and a dummy value (0.0).
        """
        lats, lons = [lat for lat,_ in locations], [lon for _,lon in locations]
        centre = self.get_variation(lats, lons)
        return [centre, 0.0] if self.data != 'Iowa' else [(41.66371064, -91.5133373), 0.0]

    def plot_locs(self, locations):
        """
        Plots the locations on a scatter plot.

        Parameters:
            locations (list[tuple]): List of locations (latitude, longitude).
        """
        lats = [lat for lat,_ in locations]
        lons = [lon for _,lon in locations]
        plt.figure(figsize=(8, 6))
        plt.scatter(lats[1:],lons[1:], color = 'r', label='Order Locations')
        plt.scatter(lats[0],lons[0], color = 'g', label='Depot Location')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title(self.data)
        plt.legend()
        plt.show()

    def plot_dist(self, times, buckets):
        """
        Plots the distribution of orders throughout the day.

        Parameters:
            times (list[int]): List of times (in minutes).
            buckets (list[int]): List of order counts per time bucket.
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Number of Orders Throughout Day ({self.epoch_length} Min Decision Epochs)')
        plt.xlabel('Time (in Minutes)')
        plt.ylabel('Number of Orders')
        plt.plot(times, buckets)
        plt.show()

    def create_dataframe(self):
        """
        Creates a DataFrame with the most frequent locations and their prevalence.

        Returns:
            pandas.DataFrame: The created DataFrame.
        """
        df = pd.read_csv(f'datasets/{self.data}/order_data.csv')
        df['coordinates'] = df.apply(lambda x: (x['latitude'], x['longitude']), axis=1)
        centre_point = self.get_centre_point(list(df.coordinates))
        occ_df = df.coordinates.value_counts()[:self.num_locations].to_frame(name='Occurences')
        new_df = [[location, occurences / sum(occ_df.Occurences)] for (location, occurences) in zip(occ_df.index, occ_df.Occurences)]
        new_df.insert(0, centre_point)
        df = pd.DataFrame(new_df, columns = ['coordinates', 'prevalence'])

        df = self.filter_and_normalize(df).reset_index(drop=True)

        return df

    def filter_and_normalize(self, df, road_speed = 20.0):
        # Extract the coordinates of the 0th index
        start_coord = df.loc[0, 'coordinates']
        
        # Calculate travel time and filter out rows that exceed max_travel_time
        df['travel_time_minutes'] = df['coordinates'].apply(lambda end_coord: 
                                                            int(round(haversine(start_coord, end_coord) / (road_speed / 60))))

        if self.bounded:
            df = df[df['travel_time_minutes'] <= self.delay_bound].copy()
        
        # Normalize prevalence values to sum up to 1
        total_prevalence = df['prevalence'].sum()
        df['prevalence'] = df['prevalence'] / total_prevalence

        # Drop the travel_time_minutes column if you don’t need it in the final output
        df = df.drop(columns=['travel_time_minutes'])
        
        return df

    def create_traveltime_file(self):
        """
        Creates a file with travel times between each pair of locations.
        """
        overall_dists = []
        for start_loc in range(len(self.df)):
            dists = []
            for end_loc in range(len(self.df)):
                start_coord = self.df.loc[start_loc]['coordinates']
                end_coord = self.df.loc[end_loc]['coordinates']
                travel_time_minutes = int(round(haversine(start_coord, end_coord) / (self.road_speed / 60)))
                if (start_loc != end_loc) and (travel_time_minutes == 0):
                    travel_time_minutes = 1
                dists.append(travel_time_minutes)
            overall_dists.append(dists)
        df = pd.DataFrame(overall_dists)
        df.to_csv(f'datasets/{self.data}/travel_time_{self.filename}.csv', index=False, header=False)

    def bucket_times(self, times):
        """
        Organizes times into buckets based on the epoch length.

        Parameters:
            times (list[int]): List of times.

        Returns:
            list[int]: List of order counts per time bucket.
        """
        buckets = {}
        times.sort()
        for time in times:
            bucket_num = time // (self.epoch_length * 60.0)
            if bucket_num not in buckets:
                buckets[bucket_num] = []
            buckets[bucket_num].append(time)
        return [len(buckets.get(i,[])) for i in range(int(1440 / self.epoch_length))]

    def get_ultra_order_distribution(self):
        """
        Retrieves and processes order data to get the distribution of orders across time.

        Returns:
            dict: A dictionary containing the distribution of orders across time.
        """
        dfs = []
        for order_file in range(1,6):
            with open(f'datasets/Ultra/orders_{order_file}.pkl', 'rb') as handle: 
                dfs.append(pickle.load(handle))
        df = pd.concat(dfs)
        times = [time for time in df.createdAtSec]
        buckets = self.bucket_times(times)
        buckets = [int(numpy.ceil(i * 0.7)) for i in buckets]
        # print(buckets)
        # exit()
        times = [t for t in range(0,1440,self.epoch_length)]
        return {time : value for time,value in zip(times, buckets)}

if __name__ == '__main__':
    # Parsing arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
    parser.add_argument('-variation_percentage', '--variation_percentage', type=float , default=0.2)
    parser.add_argument('-num_locations', '--num_locations', type=int , default=1000)
    parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
    parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
    parser.add_argument('-bounded', '--bounded', type=int , default=0)
    parser.add_argument('-delay_bound', '--delay_bound', type=int , default=10)
    parser.add_argument('-seed', '--seed', type=int , default=1)
    args = parser.parse_args()

    # Ensuring that the total minutes in a day (1440) is divisible by the epoch length
    assert not 1440 % args.epoch_length

    # Setting up the numpy random state
    np = numpy.random.RandomState(args.seed)

    # Initializing the DataPreparation class
    Prep = DataPreparation(vars(args))

    # Initializing the DataGenerator class with the prepared data
    Data = DataGenerator(data=Prep.data, 
                         variation_percentage=Prep.variation_percentage, 
                         num_locations=Prep.num_locations, 
                         road_speed=Prep.road_speed, 
                         epoch_length=Prep.epoch_length,
                         dist=Prep.dist, 
                         df=Prep.df, 
                         seed=args.seed)

    # Creating a directory for saving the generated data
    if not os.path.exists(f'generations/{Prep.filename}'):
        os.makedirs(f'generations/{Prep.filename}')

    # Saving the generated data to a file
    with open(f'generations/{Prep.filename}/data_{Prep.filename}.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

