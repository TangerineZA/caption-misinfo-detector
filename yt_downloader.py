import sys
import os.path
from pytube import YouTube

import csv
with open('FVC_text_queries.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        print("Row " + str(i))
        i = i + 1
        try:
            yt = YouTube(row[1])
            path = os.path.join('videos', row[0])
            path2 = os.path.join(path + '.mp4')
            print("Path: " + path2)
            print(path2)
            if not os.path.exists(path2):
                print(row[0] + '\n')
                video = yt.streams.first()
                print("Downloading...")
                video.download(os.path.join('videos'))
        except Exception as e:
            print("Passing on exception %s", e)
            continue