import pandas as pd 
import numpy as np 
import glob 
import os
import lightgbm

class FeatureSelector:
    """
    This class is implemented based on the idea of 
    https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Usage.ipynb.
    Use this class to select the best feature.
    Existing methods:
        * Remove missing
        * Remove single unique
        * Remove colinear
        * Remove zero importance
        * Remove low importance
    """
    def __init__(self, input_data, labels):
        self.df = input_data
        self.labels = labels
        self.columns = list(input_data.columns)
        self.num_records = len(self.df)
        pass

    def missing_cols(self, threshold=0.3):
        """
        Remove feature with missing rate larger than a 
        threhold (default 0.3). Return the list of column 
        to be removed.
        Args:
            threshold: float
                The threshold you want to filter
        Return:
            list[str]: The list of columns' name to be removed 
        """
        missing_counts = self.df.isnull().sum().tolist()
        missing_percentage = [count / self.num_records for count in missing_counts]
        missing_dict = dict(zip(self.columns, missing_percentage))
        columns_filtered = {key: value for (key, value) in missing_dict.items() 
        if value >= threshold}
        return list(columns_filtered.keys())

    def single_unique_cols(self):
        """
        Remove column with only one value
        Return:
            list[str]: list of columns with single unique value
        """
        single_unique_cols = []
        for col in self.columns:
            unique_val = self.df[col].nunique()
            if unique_val != 1 or unique_val == 0: 
                single_unique_cols.append(col) 
        return single_unique_cols

