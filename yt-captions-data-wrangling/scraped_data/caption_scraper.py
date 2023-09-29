import json
from yt_dlp import YoutubeDL
import os
import csv

ydl_opts = {
    'writesubtitles': True, 
    'writeautomaticsub': True,
    'skip_download' : True,
    'outtmpl': "C:/Users/david/Desktop/tiktec/yt-captions/subs/%(id)s",
    'subtitlesformat': 'srt'}
failed_urls = []
ydl = YoutubeDL(ydl_opts)

print(os.getcwd())

with open('.\yt-captions\youtube_audit_dataset_unique.csv', 'r', encoding='utf-8') as f:
    try:
        reader = csv.reader(f)
        for row in reader:
            try:
                ydl.download(row[4])
                print("Downloaded successfully!")
            except Exception as error:
                print('Failed to download')
                print(str(error))
                failed_urls.append(row[4])
    except Exception as error:
        print(str(error))


with open(".\yt-captions\failed_urls_vids.txt", "w") as f:
    f.writelines(failed_urls)
    
f.close()