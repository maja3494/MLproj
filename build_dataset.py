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
        # TODO
        pass

    def build_dataset(self, station_combos, begin_year, end_year, start_of_snow_year, start_of_water_year, out_path):

        for snotel_site_ids, river_gadge_ids in station_combos:

            snotel_data = build_sno_water_eq_dataset(snotel_site_ids, begin_year, end_year, start_of_snow_year)

        pass


if __name__ == '__main__':
    dsb = DatasetBuilder()
    dsb.build_dataset([
                        (['snotelid'], ['watersiteid']),
                        (['snotelid2', 'snotelid3'], ['watersiteid2', 'watersiteid3', 'watersiteid4'])
                      ],
                      '2000', '2020', '08-01', '01-01', 'dataset')
