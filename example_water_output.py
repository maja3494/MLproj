import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from matplotlib.dates import DateFormatter
from snotel_data_retriever import *


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
            y.append(float(split[-1][:-1]))
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


def plotmultiyear(filepaths):
    datas = [(fp, *ReadWaterFile(fp)) for fp in filepaths]
    fig, ax = plt.subplots(1, 1)
    for filepath, dates, water_vals in datas:
        ax.plot_date(np.arange(len(dates[90:273])), water_vals[90:273], '-', label=f"{filepath.split('/')[3][9:13]}")
    ax.set_title(f"Water level data for Boulder Creek (06730200)")
    ax.set_ylabel("Water Level (cfs)")
    ax.set_xlabel("Dates")
    ax.set_xticks(np.arange(0, len(datas[0][1][90:273]), 35))
    ax.set_xticklabels(pd.to_datetime(datas[0][1][90:273])[np.arange(0, len(datas[0][1][90:273]), 35)].strftime('%m/%d'))
    plt.legend()
    plt.show()


def OverlayData(filepath):
    dates, water_vals=ReadWaterFile(filepath)

    fig, ax = plt.subplots(1, 1)


    '''plot 1^^^'''

    '''plot 2vvv'''

    all_data = read_all_sno_water_eq_data('dataset', 'Boulder Creek', ['838', '663'])
    station_name, dataset_data = all_data[0]
    file, data = dataset_data[-2]
    ld1 = ax.plot_date(np.asarray(pd.to_datetime(data['date'])), data['value'], '-', label=f"Snow Water Eq Data")
    ax.set_ylabel("Snow water Eq (in)")
    ax.set_title(f"Boulder Run Off Data in 2019-2020 (snotel: 838, gauge: 06730200)")
    ax.set_xlabel("Dates")
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))

    ax2 = ax.twinx()
    ld2 = ax2.plot_date(dates[90:273], water_vals[90:273], '-', label=f"Water level", c='orange')
    ax2.set_ylabel("Water Level (cfs)")
    ax2.xaxis.set_major_formatter(DateFormatter('%m/%d/%y'))

    lns = ld1 + ld2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.show()


if __name__ == '__main__':
    plotmultiyear(["./WaterData/06730200/06730200_2018.csv", "./WaterData/06730200/06730200_2019.csv", "./WaterData/06730200/06730200_2020.csv"])
    OverlayData("./WaterData/06730200/06730200_2020.csv")
