import sys
import pickle
import pandas as pd


def read_data(filename) -> pd.DataFrame:
    return pd.read_parquet(filename)


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
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    model_file = 'model.bin'
    
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(filename=input_file, categorical=categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dv, lr = load_model(filename=model_file)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)