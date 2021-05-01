# from Scraper.ipynb and example_water_output
import requests
import os.path
from os import path
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import pandas as pd
import csv


def file_data_to_cleaned(data, other_num):
    lines=data.split('\n')
    x=[]
    y=[]
    for line in lines:
        if len(line) > 0 and line[0] != '#' and line[0] not in 'abcdefghijklmnopqrstuvwxyz'and line[1] not in 'as':
            split = line.split('\t')
            if other_num is not None:
                if split[3] != other_num:
                    continue
            year=split[7]
            day=split[6]
            month=split[5]
            date=year+'-'+month.zfill(2)+'-'+day.zfill(2)
            x.append(np.datetime64(date))
            y.append(float(split[-1]))

    line_data = [("i", "date", "value")] + list(zip(range(len(x)), np.asarray(pd.to_datetime(x)), y))

    return line_data


def scraper(dir_path, siteNumber, year, start_time, end_time):
    siteNumber, other_num = siteNumber
    URL=f'https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites={siteNumber}&startDT={year}-{start_time}&endDT={year}-{end_time}&statReportType=daily&statTypeCd=mean'
    page=requests.get(URL)
    soup=BeautifulSoup(page.content,'html.parser')
    results=soup.get_text()
    line_data = file_data_to_cleaned(results, other_num)
    start_datetime = datetime.strptime(f"{year}-{start_time}", '%Y-%m-%d')
    end_datetime = datetime.strptime(f"{year}-{end_time}", '%Y-%m-%d')
    fileName = f"{start_datetime.strftime('%Y%m%d')}-{end_datetime.strftime('%Y%m%d')}.csv"
    #fileName=siteNumber+"_"+year+".csv"
    # if path.exists(os.path.join(dir_path, fileName)):
    #     print('File '+os.path.join(dir_path, fileName)+' already exsists')
    # else:
    file=open(os.path.join(dir_path, fileName), "w", newline='')
    #file.write(line_data)
    csv_fw = csv.writer(file)
    csv_fw.writerows(line_data)
    file.close()
    print('File '+os.path.join(dir_path, fileName)+' created.')


def scrapeRange(dir_path, siteNumber,startYear,endYear, start_time, end_time):
    for i in range(int(startYear),int(endYear)):
        scraper(dir_path, siteNumber, str(i), start_time, end_time)


if __name__ == '__main__':
    #scrapeRange(1,2007,2019)
    siteNumber=['06730200','06727500','06736000','06752260','06751490','06719505','06716500','06700000','07096000',
               '07094500','07093700','07091200','07087050','09046490','09046600','09057500','09058000','09034250',
               '09060799','09070500','09071750','09095500','09163500','09080400','09081000','09081600','09085000']
    startYear='2007'
    endYear='2021'
    #print(os.getcwd())
    #break

    for i in siteNumber:
        #if path.exsist(i)
        os.mkdir(i)
        oldDir=os.getcwd()
        os.chdir(i)
        scrapeRange(i,startYear,endYear)
        os.chdir(oldDir)
    #print(results)