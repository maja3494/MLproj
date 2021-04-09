import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv
from snotel_data_retriever import *

class DatasetBuilder():
    def __init__(self):
        pass

    def build_dataset(self, station_combos, begin_year, end_year, start_of_snow_year, start_of_water_year, out_path):
        pass


if __name__ == '__main__':
    dsb = DatasetBuilder()
    dsb.build_dataset([('snotelid', 'watersiteid'),('snotelid', 'watersiteid'),('snotelid', 'watersiteid')],
                      '2000', '2020', '08-01', '01-01', 'snotel_data')
