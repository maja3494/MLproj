import zeep
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


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
        """
        # https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#getStations
        return self.client.service.getStations(logicalAnd=True, stateCds=state_codes, networkCds=network_codes, countyNames=county_names, stationIds=station_ids)

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
        all_station_metadata = self.get_stations_metadata_from_trips(all_station)

        county_names = list(np.unique([mdat.countyName for mdat in all_station_metadata]))

        return county_names

    def get_snow_water_eq_data(self, stations_trips, begin_date, end_date):
        """

        :param stations_trips:
        :param begin_date:
        :param end_date:
        :return: returns data for each station
        """
        # https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#getdata
        data = self.client.service.getData(stationTriplets=stations_trips, elementCd='WTEQ', ordinal=1, duration='DAILY',
                                    getFlags=True, beginDate=begin_date.strftime(self.date_str),
                                    endDate=end_date.strftime(self.date_str), alwaysReturnDailyFeb29=False)

        for i in range(len(data)):
            # convert to float
            data[i].values = [float(v) for v in data[i].values]

        return data


if __name__ == "__main__":
    # this example will get snow water equivalent data for the past year from all SnoTel stations in clear creak county
    # and plot it

    stdr = SnoTelDataRetriever()

    # all_station_counties = stdr.get_all_counties()
    # print(all_station_counties)  # for CO: ['Archuleta', 'Boulder', 'Chaffee', 'Clear Creek', 'Conejos', 'Costilla', 'Custer', 'Delta', 'Dolores', 'Eagle', 'Garfield', 'Gilpin', 'Grand', 'Gunnison', 'Hinsdale', 'Huerfano', 'Jackson', 'La Plata', 'Lake', 'Larimer', 'Las Animas', 'Mesa', 'Mineral', 'Montezuma', 'Montrose', 'Ouray', 'Park', 'Pitkin', 'Rio Blanco', 'Rio Grande', 'Routt', 'Saguache', 'San Juan', 'San Miguel', 'Summit', 'Teller']

    sntl_stations = stdr.get_all_stations(county_names=['Clear Creek'])  # station_ids=['936', '935', '602']
    sntl_stations_metadata = stdr.get_metadata_from_trips(sntl_stations)

    end = datetime.now()
    start = datetime.now()-relativedelta(years=2)

    sntl_stations_data = stdr.get_snow_water_eq_data(sntl_stations, start, end)

    fig, ax = plt.subplots(1, 1)
    for i, station in enumerate(sntl_stations_data):
        times = np.asarray(pd.to_datetime(np.linspace(pd.Timestamp(station.beginDate).value,
                                                      pd.Timestamp(station.endDate).value, len(station.values))))
        ax.plot(times, station.values, label=f"{station.stationTriplet} ({sntl_stations_metadata[i].name})")

    ax.set_title(f"Snow water Eq in Clear Creek County ({sntl_stations_data[0].beginDate[:10]} to {sntl_stations_data[0].endDate[:10]})")
    ax.set_ylabel("Snow water Eq (in)")
    ax.set_xlabel("Time")

    plt.legend()
    plt.show()
