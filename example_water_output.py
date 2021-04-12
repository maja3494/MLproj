import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as mdates

def ReadWaterFile(filepath):
    file=open(filepath,'r')
    lines=file.readlines()
    x=[]
    y=[]
    for line in lines:
        if line[0] != '#' and line[0] not in 'abcdefghijklmnopqrstuvwxyz'and line[1] not in 'as':
            split = line.split('\t')
            year=split[7]
            day=split[6]
            month=split[5]
            # print(year+'-'+month+'-0'+day+)
            date=month+'/'+day+'/'+year
            x.append(date)
            y.append(split[-1][:-1])
    dates = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in x]
    return dates, y



def plot1year(filepath):
    dates, water_vals=ReadWaterFile(filepath)
    for i, station in enumerate(sntl_stations_data):
            times = np.asarray(pd.to_datetime(np.linspace(pd.Timestamp(station.beginDate).value,
                                                          pd.Timestamp(station.endDate).value, len(station.values))))
            ax.plot_date(times, station.values, '-', label=f"{station.stationTriplet} ({sntl_stations_metadata[i].name})")

        ax.set_title(f"Snow water Eq in {county_name} County ({sntl_stations_data[0].beginDate[:10]} to "
                     f"{sntl_stations_data[0].endDate[:10]})")
        ax.set_ylabel("Snow water Eq (in)")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y')
    plt.savefig('water_level.png')

plot1year("./WaterData/06700000/06700000_2007.csv")