import os
import sys
import pickle
import pandas as pd


def get_input_path(year: int, month: int) -> str:
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename) -> pd.DataFrame:
    options = None
    
    if s3_endpoint_url := os.getenv('S3_ENDPOINT_URL'):
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }

    return pd.read_parquet(filename, storage_options=options)


def save_data(df, filename) -> None:
    options = None
    
    if s3_endpoint_url := os.getenv('S3_ENDPOINT_URL'):
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }

    df.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


def prepare_data(df, categorical) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def load_model(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    model_file = 'model.bin'
    
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(filename=input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dv, lr = load_model(filename=model_file)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())
    print('predicted sum duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)