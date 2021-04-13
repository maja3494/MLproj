import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from matplotlib.dates import DateFormatter
import read_all_sno_water_eq_data from snotel_data_retriever

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
            date=year+'-'+month.zfill(2)+'-'+day.zfill(2)
            x.append(np.datetime64(date))
            y.append(int(split[-1][:-1]))
    return x, y



def plot1year(filepath):
    dates, water_vals=ReadWaterFile(filepath)
    fig, ax = plt.subplots(1, 1)
    ax.plot_date(dates, water_vals, '-', label=f"{filepath}")
    ax.set_title(f"Water level data in {filepath}")
    ax.set_ylabel("Water Level (cfs)")
    ax.set_xlabel("Dates")
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))
    # ax.set_yticks(range(0,max(water_vals,10))
    plt.show()

def OverlayData():
    dates, water_vals=ReadWaterFile(filepath)

    fig, ax = plt.subplots(1, 1)

    
    ax.plot_date(dates, water_vals, '-', label=f"{filepath}")
    ax.plot_date(dates, water_vals, '-', label=f"{filepath}")
    ax.set_title(f"Water level data in {filepath}")
    ax.set_ylabel("Water Level (cfs)")
    ax.set_xlabel("Dates")
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))

    '''plot 1^^^'''
    '''plot 2vvv'''
    ax2=ax.twinx()
    all_data = read_all_sno_water_eq_data('dataset', 'Boulder Creek', ['838', '663'])
    station_name, dataset_data = all_data[0]
    file, data = dataset_data[-3]
    ax2.plot_date(data['date'], data['value'], '-', label=f"Snow Water Eq Data")
    ax2.set_ylabel("Snow water Eq (in)")


    plt.show()

plot1year("./WaterData/06700000/06700000_2019.csv")
OverlayData("./WaterData/06700000/06700000_2019.csv")
