import os
import zeep
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import csv


class SnoTelDataRetriever:
    def __init__(self):
        self.date_str = '%Y-%m-%d'
        self.client = zeep.Client('https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL')

    def get_all_stations(self, state_codes='CO', network_codes='SNTL', county_names=None, station_ids=None):
        """

        :param state_codes: list - optional
        :param network_codes: list - optional
        :param county_names: list - optional
        :param station_ids: list - optional
        :return: list of str with the format [station id]:[state code]:[network code] i.e "302:OR:SNTL"
        These are what we call triples or trips
        """
        # https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#getStations
        return self.client.service.getStations(logicalAnd=True, stateCds=state_codes, networkCds=network_codes,
                                               countyNames=county_names, stationIds=station_ids)

    def get_metadata_from_trips(self, stations_trips):
        """

        :param stations_trips:
        :return: metadata
        """
        # https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#getStationMetadata
        all_station_metadata = self.client.service.getStationMetadataMultiple(stationTriplets=stations_trips)

        return all_station_metadata

    def get_all_counties(self, state_codes='CO', network_codes='SNTL'):
        """

        :param state_codes:
        :param network_codes:
        :return: list of all available counties with given state_codes and network_codes
        """
        all_station = self.get_all_stations(state_codes, network_codes)
        all_station_metadata = self.get_metadata_from_trips(all_station)

        county_names = list(np.unique([mdat.countyName for mdat in all_station_metadata]))

        return county_names

    def get_snow_water_eq_data(self, stations_trips, begin_date, end_date, element_code='WTEQ'):
        """

        :param stations_trips:
        :param begin_date:
        :param end_date:
        :param element_code: str - optional
        :return: returns data for each station
        """
        # https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#getdata
        data = self.client.service.getData(stationTriplets=stations_trips, elementCd=element_code, ordinal=1,
                                    duration='DAILY', getFlags=True, beginDate=begin_date.strftime(self.date_str),
                                    endDate=end_date.strftime(self.date_str), alwaysReturnDailyFeb29=False)  # TODO: do we want this?

        for i in range(len(data)):
            # convert to float
            data[i].values = [float(v) for v in data[i].values]

        return data


def build_sno_water_eq_dataset(station_ids, begin_year, end_year, start_of_snow_water_year):
    if isinstance(station_ids, str):
        station_ids = [station_ids]

    stdr = SnoTelDataRetriever()
    sntl_stations = stdr.get_all_stations(station_ids=station_ids)
    sntl_stations_metadata = stdr.get_metadata_from_trips(sntl_stations)
    assert len(sntl_stations_metadata) == len(station_ids)

    req_begin_time = datetime.strptime(f"{begin_year}-{start_of_snow_water_year}", '%Y-%m-%d')
    end_time = datetime.strptime(f"{end_year}-{start_of_snow_water_year}", '%Y-%m-%d') + relativedelta(years=1)

    start_dates = []
    station_names = []
    for station_metadata in sntl_stations_metadata:
        station_names.append(station_metadata.name)
        dt_earlest_for_station = datetime.strptime(station_metadata.beginDate, '%Y-%m-%d %H:%M:%S')
        dt = datetime(dt_earlest_for_station.year, req_begin_time.month, req_begin_time.day)
        if dt < dt_earlest_for_station:
            dt += relativedelta(years=1)

        start_dates.append(max(req_begin_time, dt))

    start_time = max(start_dates)

    station_dataset = [[] for _ in range(len(sntl_stations))]

    # todo: should we out all data into the same file
    datas = [None for _ in range(len(sntl_stations))]
    this_start_time = start_time
    while this_start_time < end_time:
        this_end_date = this_start_time + relativedelta(years=1)

        sntl_data = stdr.get_snow_water_eq_data(sntl_stations, this_start_time, this_end_date)

        for i in range(len(sntl_data)):
            times = np.asarray(pd.to_datetime(np.linspace(pd.Timestamp(sntl_data[i].beginDate).value,
                                                          pd.Timestamp(sntl_data[i].endDate).value, len(sntl_data[i].values))))
            # times = [str(time_v)[:10] for time_v in times]

            data = sntl_data[i].values

            row_list = [("i", "date", "value")] + list(zip(range(len(sntl_data[i].values)), times, data))

            station_dataset[i].append((this_start_time, this_end_date, row_list))

        this_start_time = this_end_date

    return station_dataset


def read_all_sno_water_eq_data(dataset_path, station_ids):
    full_dataset = []
    sub_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    sub_dirs = [d for d in sub_dirs if d in station_ids]
    for sub_dir in sub_dirs:
        data_files = [d for d in os.listdir(os.path.join(dataset_path, sub_dir)) if os.path.splitext(d)[1] == '.csv']
        data = [(data_file, pd.read_csv(os.path.join(dataset_path, sub_dir, data_file))) for data_file in data_files]
        full_dataset.append((sub_dir, data))

    return full_dataset


if __name__ == "__main__":
    if False:
        # will build dataset, read from dataset and then plot
        # this line only needs to be run once
        data = build_sno_water_eq_dataset(['838', '663'], '1960', '2020', '08-01')

        # todo save to csv: the next line doesn't work anymore

        station_name, dataset_data = read_all_sno_water_eq_data('snotel_data', ['838', '663'])[0]

        fig, ax = plt.subplots(1, 1)
        for file, data in dataset_data[:-1]:
            # pd.DatetimeIndex(data['date']).year  # gets years of each value
            ax.plot(np.arange(len(data['value'])), data['value'], '-', label=file)

        ax.plot(np.arange(len(dataset_data[-1][1]['value'])), dataset_data[-1][1]['value'].values, '-', label=dataset_data[-1][0], linewidth=3.0)

        ax.set_title(f"Snow water Eq for {station_name}")
        ax.set_ylabel("Snow water Eq (in)")
        ax.set_xlabel("Time")
        #ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))

        plt.legend()
        plt.show()

    ####################################################################################################################

    if False:
        # this example will get snow water equivalent data for the past year from all SnoTel stations in clear creak county
        # and plot it
        stdr = SnoTelDataRetriever()

        # all_station_counties = stdr.get_all_counties()
        # print("Select County:\n", list(enumerate(all_station_counties)))  # for CO: ['Archuleta', 'Boulder', 'Chaffee', 'Clear Creek', 'Conejos', 'Costilla', 'Custer', 'Delta', 'Dolores', 'Eagle', 'Garfield', 'Gilpin', 'Grand', 'Gunnison', 'Hinsdale', 'Huerfano', 'Jackson', 'La Plata', 'Lake', 'Larimer', 'Las Animas', 'Mesa', 'Mineral', 'Montezuma', 'Montrose', 'Ouray', 'Park', 'Pitkin', 'Rio Blanco', 'Rio Grande', 'Routt', 'Saguache', 'San Juan', 'San Miguel', 'Summit', 'Teller']
        # county_num = int(input(f"Counties: {list(enumerate(all_station_counties))}\nSelect number: "))
        # county_name = all_station_counties[county_num]  # 'Summit'
        county_name = 'Clear Creek'
        print("Using county:", county_name)

        sntl_stations = stdr.get_all_stations(county_names=[county_name])  # station_ids=['936', '935', '602']
        sntl_stations_metadata = stdr.get_metadata_from_trips(sntl_stations)

        end = datetime.now()
        start = datetime.now()-relativedelta(years=2)

        sntl_stations_data = stdr.get_snow_water_eq_data(sntl_stations, start, end)

        fig, ax = plt.subplots(1, 1)
        for i, station in enumerate(sntl_stations_data):
            times = np.asarray(pd.to_datetime(np.linspace(pd.Timestamp(station.beginDate).value,
                                                          pd.Timestamp(station.endDate).value, len(station.values))))
            ax.plot_date(times, station.values, '-', label=f"{station.stationTriplet} ({sntl_stations_metadata[i].name})")

        ax.set_title(f"Snow water Eq in {county_name} County ({sntl_stations_data[0].beginDate[:10]} to "
                     f"{sntl_stations_data[0].endDate[:10]})")
        ax.set_ylabel("Snow water Eq (in)")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))

        plt.legend()
        plt.show()

    ####################################################################################################################

    if True:
        stdr = SnoTelDataRetriever()

        all_station_counties = stdr.get_all_counties()
        #print("Select County:\n", list(enumerate(all_station_counties)))  # for CO: ['Archuleta', 'Boulder', 'Chaffee', 'Clear Creek', 'Conejos', 'Costilla', 'Custer', 'Delta', 'Dolores', 'Eagle', 'Garfield', 'Gilpin', 'Grand', 'Gunnison', 'Hinsdale', 'Huerfano', 'Jackson', 'La Plata', 'Lake', 'Larimer', 'Las Animas', 'Mesa', 'Mineral', 'Montezuma', 'Montrose', 'Ouray', 'Park', 'Pitkin', 'Rio Blanco', 'Rio Grande', 'Routt', 'Saguache', 'San Juan', 'San Miguel', 'Summit', 'Teller']
        #county_num = int(input(f"Counties: {list(enumerate(all_station_counties))}\nSelect number: "))

        for county_name in all_station_counties:
            sntl_stations = stdr.get_all_stations(county_names=[county_name])
            print("county_name:", county_name, "sntl_stations:", [sntl_station.split(':')[0] for sntl_station in sntl_stations])

            """
            county_name: Archuleta sntl_stations: ['1160']
            county_name: Boulder sntl_stations: ['412', '663', '1251', '838', '1042']
            county_name: Chaffee sntl_stations: ['1100']
            county_name: Clear Creek sntl_stations: ['936', '935', '602']
            county_name: Conejos sntl_stations: ['431', '580']
            county_name: Costilla sntl_stations: ['430', '829', '1005']
            county_name: Custer sntl_stations: ['773']
            county_name: Delta sntl_stations: ['675', '682']
            county_name: Dolores sntl_stations: ['1185', '465', '586', '739']
            county_name: Eagle sntl_stations: ['1041', '1040', '842']
            county_name: Garfield sntl_stations: ['345', '827']
            county_name: Gilpin sntl_stations: ['564']
            county_name: Grand sntl_stations: ['1030', '305', '335', '913', '1186', '1187', '970', '565', '1014', '688', '793', '869']
            county_name: Gunnison sntl_stations: ['380', '669', '680', '701', '737', '1141']
            county_name: Hinsdale sntl_stations: ['762', '839', '1188']
            county_name: Huerfano sntl_stations: ['303']
            county_name: Jackson sntl_stations: ['408', '1031', '1032', '1033']
            county_name: La Plata sntl_stations: ['904', '843']
            county_name: Lake sntl_stations: ['369']
            county_name: Larimer sntl_stations: ['322', '1161', '438', '1122', '551', '1123', '718', '870']
            county_name: Las Animas sntl_stations: ['857']
            county_name: Mesa sntl_stations: ['622']
            county_name: Mineral sntl_stations: ['327', '624', '840', '874']
            county_name: Montezuma sntl_stations: ['905', '1060']
            county_name: Montrose sntl_stations: ['409']
            county_name: Ouray sntl_stations: ['538']
            county_name: Park sntl_stations: ['938', '937', '939']
            county_name: Pitkin sntl_stations: ['1101', '542', '547', '556', '618', '658']
            county_name: Rio Blanco sntl_stations: ['378', '426', '717']
            county_name: Rio Grande sntl_stations: ['1058']
            county_name: Routt sntl_stations: ['1061', '457', '467', '1252', '940', '607', '709', '825']
            county_name: Saguache sntl_stations: ['1059', '1102', '914', '1124', '1128']
            county_name: San Juan sntl_stations: ['386', '387', '629', '632', '713', '780', '797']
            county_name: San Miguel sntl_stations: ['589']
            county_name: Summit sntl_stations: ['415', '1120', '485', '505', '531', '802']
            county_name: Teller sntl_stations: ['1057']
            """

