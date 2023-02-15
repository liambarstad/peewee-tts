import sys
import boto3
import json
from io import StringIO

client = boto3.client('lambda')

num_workers = int(sys.argv[1])


for i in range(num_workers):
    payload = StringIO()
    json.dump({
        'worker_index': i,
        'total_number': num_workers
    }, payload)

    res = client.invoke(
        FunctionName='extract_tarfile_to_s3',
        InvocationType='Event',
        Payload=payload.getvalue()
    )
    print(f'Worker {i} invoked, response: {res}')
