
import os
import pandas as pd
from datetime import datetime
from pandas.testing import assert_frame_equal
from batch import read_data, save_data, get_output_path, get_input_path

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_s3_prediction():
    # Arrange
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    df_input = pd.DataFrame(data, columns=columns)

    filename = get_input_path(2023, 1)
    save_data(df_input, filename)

    # Act
    os.system("python batch.py 2023 1")

    filename = get_output_path(2023, 1)
    df_actual = read_data(filename)

    # Assert
    columns = ['ride_id', 'predicted_duration']
    data = [
        ('2023/01_0', 23.197149),
        ('2023/01_1', 13.080101),
    ]
    df_expected = pd.DataFrame(data, columns=columns)

    assert_frame_equal(df_actual, df_expected)