import sklearn
import scipy
import pandas as pd
import numpy as np

from DataCollector import DataCollector

if __name__ == "__main__":
    dc = DataCollector(name='lab1', data_source='lab1_data.csv')
    dc.collect_data()
    dc.write_data_source()
    print("Data collection and saving completed. Have a nice day!")