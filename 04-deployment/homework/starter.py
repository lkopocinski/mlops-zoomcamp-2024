import click
import pickle
import pandas as pd

CATEGORICAL = ['PULocationID', 'DOLocationID']

def load_model(path: str):
    with open(path, 'rb') as f_in:
        return pickle.load(f_in)

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df


@click.command()
@click.option('--year')
@click.option('--month')
def main(year: str, month: str):
    dv, model = load_model('model.bin')
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    df['y_pred'] = y_pred

    print(f"Standard deviation of the predicted duration is: {y_pred.std():.3f}")
    print(f"Mean of the predicted duration is: {y_pred.mean():.3f}")

    df_result = df[['ride_id', 'y_pred']]

    df_result.to_parquet(
        "predictions.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    main()