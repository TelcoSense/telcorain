import os

import numpy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.compat.v1.keras.backend import set_session


CNN_OUTPUT_LEFT_NANS_LENGTH = 327


# Limit GPU memory usage to avoid processes to run out of memory.
# For a list of processes blocking GPU memory on an nvidia GPU type 'nvidia-smi' in the terminal.
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "0"
set_session(tf.compat.v1.Session(config=config))


def get_model_file_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "cnn_model_files")


modelh5_fn = str(get_model_file_path() + "/model_2020.002.180m.h5")
modeljson_fn = str(get_model_file_path() + "/model_2020.002.180m.json")

# load json and create model
json_file = open(modeljson_fn, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(modelh5_fn)
model.compile()


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=False
    )


def _add_right_padding(arr: numpy.ndarray, pads: int):
    last_val_repeated = np.repeat(arr[-1], pads)
    new_arr = np.concatenate((arr[pads:], last_val_repeated))
    return new_arr


def _add_left_padding(arr, pads: int):
    first_val_repeated = np.repeat(arr[0], pads)
    new_arr = np.concatenate((first_val_repeated, arr))[:len(arr)]
    return new_arr


def cnn_wet_dry(
    trsl_channel_1,
    trsl_channel_2,
    threshold=None,
    batch_size=100,
    verbose=0,
):
    """
    Wet dry classification using the CNN based on channel 1 and channel 2 of a CML

    Parameters
    ----------
    trsl_channel_1 : iterable of float
         Time series of received signal level of channel 1
    trsl_channel_2 : iterable of float
         Time series of received signal level of channel 2
    threshold : float or None
         Threshold between 0 and 1 which has to be surpassed to classify a period as 'wet'.
         If None, then no threshold is applied and raw wet probabilities in the range of [0,1] are returned.
    batch_size : int
        Batch size for parallel computing. Set to 1 when using a CPU!
    verbose : int
        Toggles Keras text output during prediction. Default is off.


    Returns
    -------
    iterable of int
        Time series of wet/dry classification

    Note
    ----
    Implementation of CNN method [1]_

    References
    ----------
    .. [1] Polz, J., Chwala, C., Graf, M., and Kunstmann, H.: Rain event detection in commercial microwave link
      attenuation data using convolutional neural networks, Atmos. Meas. Tech., 13, 3835–3853,
      https://doi.org/10.5194/amt-13-3835-2020, 2020.

    """

    #################
    # Normalization #
    #################
    df = pd.DataFrame()
    df["trsl1"] = _add_right_padding(trsl_channel_1.copy(), 29)
    df["trsl2"] = _add_right_padding(trsl_channel_2.copy(), 29)
    df["med1"] = df["trsl1"].rolling(72 * 60, min_periods=2 * 60, center=False).median()
    df["med2"] = df["trsl2"].rolling(72 * 60, min_periods=2 * 60, center=False).median()
    df["trsl1"] = df["trsl1"].sub(df["med1"])
    df["trsl2"] = df["trsl2"].sub(df["med2"])
    # replace NaN by -9999 during processing
    df = df.fillna(value=-9999)

    #########################
    # generate numpy arrays #
    #########################

    x_fts = np.moveaxis(
        np.array(
            [
                _rolling_window(df["trsl1"].values, 180),
                _rolling_window(df["trsl2"].values, 180),
            ]
        ),
        source=0,
        destination=-1,
    )
    cnn_pred = np.ravel(model.predict(x_fts, batch_size=batch_size, verbose=verbose))

    # set prediction to NaN whenever -9999 occurs in the moving window of one channel
    for i in range(len(cnn_pred)):
        if (
            -9999 in df["trsl1"].values[i : i + 180]
            or -9999 in df["trsl2"].values[i : i + 180]
        ):
            cnn_pred[i] = np.nan

    # Due to the moving window approach predictions can only be made for minutes 181 to N.
    # The following line fills beginning of the prediction array with NaNs.
    padded_cnn_pred = _add_left_padding(cnn_pred, 29)
    df["prediction"] = np.concatenate(
        (np.repeat(np.nan, 179), padded_cnn_pred), axis=0
    )

    if threshold is None:
        return df.prediction.values.reshape(len(trsl_channel_1))

    else:
        # only apply threshold to non NaN values
        pred = df.prediction.values
        pred[~np.isnan(pred)] = pred[~np.isnan(pred)] > threshold
        return pred
