class RequestOrder(object):
    """
    A class representing a request order, encapsulating details like destination, 
    origin time, and deadline.

    Attributes:
        destination (tuple): The destination coordinates of the order.
        origin_time (int): The time at which the order was made.
        deadline (int): The deadline time for the order. Initially set to -1.
        id (int): The identifier for the order. Initially set to -1.
        state (list): List containing the current state of the order.
        state_str (str): String representation of the order's current state.
    """

    def __init__(self, current_time, destination):
        """
        Constructor for the RequestOrder class.

        Parameters:
            current_time (int): The time at which the order is made.
            destination (tuple): The destination coordinates of the order.
        """
        self.destination = destination
        self.origin_time = current_time
        self.deadline = -1
        self.id = -1
        self.state = None
        self.state_str = None

    def __str__(self):
        """
        String representation of the RequestOrder.

        Returns:
            str: A string describing the order, including its ID, destination, and deadline.
        """
        return (f'Order {self.id} ({self.destination}, {self.deadline})')

    def __repr__(self):
        """
        The official string representation of the RequestOrder, used for debugging.

        Returns:
            str: A string representation of the RequestOrder object.
        """
        return str(self)

    def update_state(self, current_time):
        """
        Updates the state of the order based on the current time.

        Parameters:
            current_time (int): The current time to update the order state with.
        """
        self.state = [self.destination, self.deadline, current_time]
        self.state_str = str(self.state)
