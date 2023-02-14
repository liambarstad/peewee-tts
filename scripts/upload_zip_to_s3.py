import os
import sys
import math
import time
import boto3
import tarfile
import threading
import pickle
import pandas as pd

THREADS = 1

s3 = boto3.client('s3')

tf = tarfile.open('data/utterance_corpuses/train-other-500.tar.gz', 'r:gz')

paths = []

if os.path.isfile('scripts/paths.pkl'):
    with open('scripts/paths.pkl', 'rb') as ff:
        paths = pickle.load(ff)
else:
    paths = [ path for path in tf ]
    with open('scripts/paths.pkl', 'wb') as ff:
        pickle.dump(paths, ff)

paths_series = pd.Series([ path.name.split('./')[1] for path in paths if path.name.startswith('./') ])
print('paths parsed from file')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=os.environ['BUCKET_NAME'])
existing_files = [ f['Key'] for page in pages for f in page['Contents']]
existing_files_series = pd.Series(existing_files)

todo_inds = paths_series[~paths_series.isin(existing_files_series)].index
todo_paths = [ paths[x] for x in todo_inds ]

counts = [ 0 for _ in range(THREADS) ]

def upload_from_tf(paths, tf, counts, ind, total_count):
    print(f'thread {ind+1} executing')
    local_client = boto3.client('s3')
    for path in paths:
        try:
            start_ts = time.time()
            file = tf.extractfile(path)
            tar_ts = time.time()
            if file:
                filename = path.name.split('./')[1]
                local_client.upload_fileobj(
                    file, 
                    os.environ['BUCKET_NAME'],
                    filename
                )
                counts[ind] += 1
                end_ts = time.time()
                print(' | '.join([
                    f'thread {i+1}: {c}'
                    for i, c in enumerate(counts)
                ]) + f' | total: {sum(counts)}/{total_count}' 
                + ' | time for parsing: ' + str(round(end_ts - start_ts, 2)) + 'sec'
                + ' | time for tar extract: ' + str(round(tar_ts - start_ts, 2)) + 'sec'
                + ' | time for upload: ' + str(round(end_ts - tar_ts, 2)) + 'sec' 
                )
        except Exception as e:
            print('ERR THREAD '+str(ind)+':'+str(e))
            continue

threads = []
partition = math.ceil(len(todo_paths) / THREADS)

for ind in range(THREADS):
    partitioned_paths = todo_paths[ind*partition:(ind+1)*partition] \
            if ind + 1 < THREADS \
            else todo_paths[ind*partition:]
    print(f'thread {ind+1} todo: {len(partitioned_paths)}')
    thread = threading.Thread(target=upload_from_tf, args=(partitioned_paths, tf, counts, ind, len(todo_paths)))
    thread.setDaemon(True)
    threads.append(thread)
    thread.start()

proccesses_alive = True

print('......................................:)')
while proccesses_alive:
    proccesses_alive = False
    for thread in threads:
        if thread.is_alive():
            proccesses_alive = True
    time.sleep(180)

print('ALL DONE')
