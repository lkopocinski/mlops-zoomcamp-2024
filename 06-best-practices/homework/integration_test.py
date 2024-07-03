
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_s3():
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    df_input = pd.DataFrame(data, columns=columns)
    
    options = {
        'client_kwargs': {
            'endpoint_url': "http://127.0.0.1:4566"
        }
    }

    output_pattern = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    input_file = output_pattern.format(year=2023, month=1)

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

if __name__ == '__main__':
    test_s3()