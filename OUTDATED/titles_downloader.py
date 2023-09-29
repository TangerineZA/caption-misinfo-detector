import csv
import json
from yt_dlp import YoutubeDL

ydl = YoutubeDL()

titles_dict = {}
with open('FVC_text_queries.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            info_dict = ydl.extract_info(row[1], download=False)
            video_title = info_dict.get('title', None)
            titles_dict[row[1]] = video_title
            print(video_title)
        except:
            print("Error")

print(titles_dict)

f.close()

with open("title_dict.txt", 'w') as f:
    json.dump(titles_dict, f)
    print("Dump completed!")
f.close()