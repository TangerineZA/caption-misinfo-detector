import json
from yt_dlp import YoutubeDL

import csv
ydl_opts = {
    'format' : 'mp4',
    'writesubtitles': True, 
    'writeautomaticsub': True}
failed_urls = []
ydl = YoutubeDL(ydl_opts)

with open('FVC_text_queries.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            ydl.download(row[1])
            print("Downloaded successfully!")
        except Exception as error:
            print('Failed to download')
            failed_urls.append(row[1])


with open("failed_urls.txt", "w") as f:
    f.writelines(failed_urls)
    
f.close()