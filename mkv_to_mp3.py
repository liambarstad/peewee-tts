import moviepy.editor as mp

def extract(vid_path, dest_path):
    mp_audio = mp.AudioFileClip(vid_path)
    mp_audio.write_audiofile(f'{dest_path}')

extract("./data/Pee-wee's Playhouse S01E01 Ice Cream Soup.mkv", './data/pee_wees_playhouse_01_01.mp3')
