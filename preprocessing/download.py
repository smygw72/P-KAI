import os
import subprocess
import pandas as pd


def trim(row, output_dir):
    original_data_path = f'{output_dir}/{row["ID"]}.mp3'
    trimmed_data_path = f'{output_dir}/{row["filename"]}.mp3'

    if not os.path.isfile(original_data_path):
        return

    start_time = str(row[2]).split(':')
    end_time = str(row[3]).split(':')

    start_sec = 60 * float(start_time[0]) + float(start_time[1])
    end_sec = 60 * float(end_time[0]) + float(end_time[1])
    duration = end_sec - start_sec

    subprocess.call([
        'ffmpeg',
        '-ss', str(start_sec),
        '-t', str(duration),
        '-i', original_data_path,
        trimmed_data_path
    ])
    # os.remove(original_data_path)


def download(row, output_dir):

    original_data_path = f'{output_dir}/{row["ID"]}.mp3'
    if os.path.isfile(original_data_path):
        return

    # url = f'https://www.youtube.com/watch?v={data_id}'
    cmd = f'youtube-dl -x -o {original_data_path} --audio-format mp3 https://www.youtube.com/watch?v={row["ID"]}'

    subprocess.run(cmd, shell=True)


def main(*args, **kwargs):
    df = pd.read_csv(f'./annotation/youtube.csv', header=0)
    # df.drop_duplicates(subset='ID', inplace=True)
    os.makedirs(f'../dataset/', exist_ok=True)

    for i, row in df.iterrows():
        print(row)
        data_path = f'../dataset/{row["filename"]}.mp3'
        if not os.path.isfile(data_path):
            download(row, f'../dataset/')
            trim(row, f'../dataset/')


if __name__ == "__main__":
    main()
