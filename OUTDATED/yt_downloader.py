import sys
import os.path
from pytube import YouTube
import json

import csv
truth_dict = dict()
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
                video = yt.streams.filter(file_extension="mp4").first()
                print("Downloading...")
                print(video.title + " - " + str(row[2]))
                video.download(os.path.join('videos'))
                truth_dict[video.title] = row[2]
                
        except Exception as e:
            print("Passing on exception %s", e)
            continue

with open("truth-dict.txt", "w") as f:
    json.dump(truth_dict, f)
    print("Saved truth dict!")