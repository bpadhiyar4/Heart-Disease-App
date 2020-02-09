"""
This module will consist important classes for the project
"""

import pandas as pd
import numpy as np

class Learning():
    """
        This is the class based implementation of project of heart disease analysis
    """
    def __init__(self):
        """
        Constructor with load the data from csv
        """
        self.data = pd.read_csv("dataset/heart.csv")

    def getData(self):
        """
        return a array of dataframe
        """
        return np.array(self.data).reshape((-1,14))
