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
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for basin_name, snotel_site_ids, river_gauge_ids in station_combos:
            dir_path = os.path.join(out_path, basin_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            snotel_data = build_sno_water_eq_dataset(snotel_site_ids, begin_year, end_year, start_of_snow_year)
            river_gauge_data = []

            snotel_dir_path = os.path.join(dir_path, 'snotel')
            river_gauge_dir_path = os.path.join(dir_path, 'water_gauge')
            if not os.path.exists(snotel_dir_path):
                os.makedirs(snotel_dir_path)
            if not os.path.exists(river_gauge_dir_path):
                os.makedirs(river_gauge_dir_path)

            for i, snotel_cite_data in enumerate(snotel_data):
                snotel_id = snotel_site_ids[i]

                site_dir_path = os.path.join(snotel_dir_path, str(snotel_id))
                if not os.path.exists(site_dir_path):
                    os.makedirs(site_dir_path)

                for start_time, end_date, snotel_csv_row_data in snotel_cite_data:
                    file_path = os.path.join(site_dir_path,
                            f"{start_time.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv")

                    with open(file_path, 'w', newline='') as fw:
                        csv_fw = csv.writer(fw)
                        csv_fw.writerows(snotel_csv_row_data)

            for i, river_gauge_cite_data in enumerate(river_gauge_data):
                river_gauge = river_gauge_ids[i]

                pass  # TODO


if __name__ == '__main__':
    dsb = DatasetBuilder()
    dsb.build_dataset([
                        ('Boulder Creek', ['838', '663'], ['06727500'])  #,
                        #('basin_name', ['snotelid2', 'snotelid3'], ['watersiteid2', 'watersiteid3', 'watersiteid4'])
                      ],
                      '2000', '2020', '08-01', '01-01', 'dataset')
