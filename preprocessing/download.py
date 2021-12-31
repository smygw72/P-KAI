import os
import subprocess
import pandas as pd

from config.config import CONFIG


def trim(row, output_dir):
    original_data_path = f'{output_dir}/{row["ID"]}_original.mp3'
    trimmed_data_path = f'{output_dir}/{row["ID"]}.mp3'

    if not os.path.isfile(original_data_path):
        return

    start_time = str(row[1]).split(':')
    end_time = str(row[2]).split(':')

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
    os.remove(original_data_path)


def download(row, output_dir):

    original_data_path = f'{output_dir}/{row["ID"]}_original.mp3'
    if os.path.isfile(original_data_path):
        return

    data_id = row['ID']
    url = f'https://www.youtube.com/watch?v={data_id}'
    subprocess.check_output([
        'youtube-dl', url,
        '-i',  # ダウンロードエラーを無視
        '-o', f'{output_dir}/%(id)s_original.%(ext)s',
        '-x', '--audio-format', 'mp3'  # 音声のみ
    ])


def main(*args, **kwargs):
    df = pd.read_csv(f'./annotation/{CONFIG.data.target}/youtube.csv', header=0)
    df.drop_duplicates(subset='ID', inplace=True)
    os.makedirs(f'../dataset/sounds', exist_ok=True)

    for i, row in df.iterrows():
        print(row)
        data_path = f'../dataset/sounds/{row["ID"]}.mp3'
        if not os.path.isfile(data_path):
            download(row, f'../dataset/sounds')
            trim(row, f'../dataset/sounds')


if __name__ == "__main__":
    main()