import os
import csv

with open('dummy.csv', 'w') as csvfile : 
    csvwriter = csv.DictWriter(csvfile, fieldnames=['perf_key', 'plyrid', 'seg_start', 'seg_duration'])
    csvwriter.writeheader()

    for i in range(30):
        for j in range(10): 
            for k in range(10): 
                csvwriter.writerow(
                        {'perf_key' :  str(i * 10 + j) ,
                        'plyrid': str(i),
                        'seg_start': k*3,
                        'seg_duration': 3.0})

