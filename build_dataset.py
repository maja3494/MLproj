import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv
from snotel_data_retriever import *
from usgs_scraper import *


class DatasetReader():
    def __init__(self, dataset_dir, basin_name, snotel_station, water_gauge, loops=10, year_range=(1960,2016)):
        self.dataset_dir = dataset_dir
        self.snotel_data_dir = os.path.join(dataset_dir, basin_name, 'snotel', snotel_station)
        self.water_gauge_dir = os.path.join(dataset_dir, basin_name, 'water_gauge', water_gauge)

        self.snotel_files = [os.path.join(self.snotel_data_dir, d) for d in os.listdir(self.snotel_data_dir) if os.path.splitext(d)[1] == '.csv']
        self.water_gauge_files = [os.path.join(self.water_gauge_dir, d) for d in os.listdir(self.water_gauge_dir) if os.path.splitext(d)[1] == '.csv']

        self.file_pairs = []
        for stf in self.snotel_files:
            stf_bn = os.path.basename(stf)
            if year_range[0] < int(stf_bn[9:13]) <= year_range[1]:
                wgfs = [wgf for wgf in self.water_gauge_files if os.path.basename(wgf).find(stf_bn[9:13]) == 0]
                if len(wgfs) > 0:
                    self.file_pairs.append((stf, wgfs[0]))

        self.loops = loops

    def __iter__(self):
        assert len(self.file_pairs) > 0
        self.i = 0

        return self

    def __next__(self):
        if self.i >= len(self.file_pairs)*self.loops:
            raise StopIteration

        i_p = self.i % len(self.file_pairs)

        if i_p == 0:
            self.p = np.random.permutation(len(self.file_pairs))

        snotel_file, water_gauge_file = self.file_pairs[self.p[i_p]]
        snotel_data = pd.read_csv(snotel_file)
        water_gauge_data = pd.read_csv(water_gauge_file)

        look_back = int(np.clip(np.random.normal(25, 50, 1),0,100)[0])

        # todo: read datas
        self.i+=1

        return snotel_data['value'].values[:len(snotel_data['value'].values)-look_back], water_gauge_data['value'].values


class DatasetBuilder():
    def __init__(self):
        # TODO
        pass

    def build_dataset(self, station_combos, begin_year, end_year, snow_start_end, water_start_end, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for basin_name, snotel_site_ids, river_gauge_ids in station_combos:
            dir_path = os.path.join(out_path, basin_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            snotel_data = build_sno_water_eq_dataset(snotel_site_ids, begin_year, end_year, *snow_start_end)
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
                    print(f"File {file_path} created")

            # for i, river_gauge_cite_data in enumerate(river_gauge_data):
            #     river_gauge = river_gauge_ids[i]
            #
            #     pass  # TODO

            for river_gauge_id in river_gauge_ids:

                site_dir_path = os.path.join(river_gauge_dir_path, str(river_gauge_id))
                if not os.path.exists(site_dir_path):
                    os.makedirs(site_dir_path)

                scrapeRange(site_dir_path, river_gauge_id, str(int(begin_year)+1), end_year, water_start_end[0], water_start_end[1])


if __name__ == '__main__':
    dsb = DatasetBuilder()
    dsb.build_dataset([
                        ('Boulder Creek', ['838', '663'], ['06730200'])  #,
                        #('basin_name', ['snotelid2', 'snotelid3'], ['watersiteid2', 'watersiteid3', 'watersiteid4'])
                      ], '1986', '2021', ('10-01', '06-30'), ('01-01', '09-30'), 'dataset')

    dsr = DatasetReader('dataset', 'Boulder Creek', '663', '06730200', 1, (1960, 2013))
    datas_train = list(dsr)
    print(len(datas_train))

    dsr = DatasetReader('dataset', 'Boulder Creek', '663', '06730200', 1, (2013, 2050))
    datas_test = list(dsr)
    print(len(datas_test))
