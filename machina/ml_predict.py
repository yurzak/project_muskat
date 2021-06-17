# -*- coding: utf-8 -*-
"""
Module for predictions based on Machine learning model
"""
import numpy as np
import pandas as pd

import time, datetime
import logging
import pickle
import os

from multiprocessing import Queue
import queue

from machina import data_utils as du
from machina import tech_analysis as ta

ML_ENABLED = 0

if ML_ENABLED:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = logging.getLogger(__name__)
QUEUE_RATE = 0.1
MARKET_OPEN_HOUR = 10

IS_NEURAL = 1
BUY_LIMIT = 0.3
CONF_LEVEL = 50
MODEL_FILE = "model_A64_T02-1615026046"


class MachineCore(object):
    def __init__(self, inQ: Queue, outQ: Queue):
        self.inQ = inQ  # raw data from IBAPI
        self.outQ = outQ  # processed data to GUI
        self.model_opened = False
        self.exitReq = False
        self.process_rate = 0

        self.scaler = None
        self.model = None
        self.features = ['Open_change', 'High_change', 'Low_change', 'Close_change', 'Last_row',
                         'Last5_ratio', 'Last10_ratio',
                         'OpenDay_dist', 'HighDay_dist', 'LowDay_dist', 'Vwap_dist',
                         'Ema10_dist', 'Ema50_dist', 'Volume_change',
                         'Trades_change', 'VolumeEma10_dist', 'Tr', 'Time', 'isBuy']

        self.data_1min = pd.DataFrame()
        self.last_predictions = []

    def start(self):
        logging.info("Running MLCore main loop")
        while not self.exitReq:
            self.readQ()
        logging.info("Finished MLCore main loop")

    def readQ(self):
        """Read Queue (inQ) method and call corresponding handler method"""
        try:
            q = self.inQ.get(timeout=QUEUE_RATE)
            #q = self.inQ.get_nowait()
        except queue.Empty:
            pass
        else:
            request = q[0]
            t1 = time.perf_counter()

            if request == "DATA_UPDATE" and ML_ENABLED and self.model_opened:
                self.on_data_update(q)
            elif request == "DATA_END" and ML_ENABLED and self.model_opened:
                self.on_data_end(q)
            elif request == "OPEN_MODEL" and ML_ENABLED and not self.model_opened:
                self.open_model()
            elif request == "EXIT":
                self.exitReq = True

            t2 = time.perf_counter()
            self.process_rate = 1 / (t2 - t1)
            #print("MLCore prediction rate:", round(self.process_rate, 3))

    def open_model(self):

        scalerfile = "scaler.sav"
        modelfile = "model.sav"
        featuresfile = "features.sav"
        #print(os.getcwd())
        if not self.model_opened:
            try:
                self.scaler = pickle.load(open(os.getcwd() + f"\\machina\\models\\" + MODEL_FILE + "\\" + scalerfile, "rb"))
                self.features = pickle.load(open(os.getcwd() + f"\\machina\\models\\" + MODEL_FILE + "\\" + featuresfile, "rb"))

                if IS_NEURAL:
                    self.model = tf.keras.models.load_model(os.getcwd() + f"\\machina\\models\\" + MODEL_FILE)
                else:
                    self.model = pickle.load(open(os.getcwd() + f"\\machina\\models\\" + MODEL_FILE + "\\" + modelfile, "rb"))

                self.model_opened = True
            except:
                print("Error with MLCore model access")
                logging.error("Error with MLCore model access")
                self.model_opened = False

    def on_data_update(self, q: Queue):

        column_list_in = ["Time", "Open", "High", "Low", "Close", "Volume", "Count", "Wap"]
        self.data_1min = pd.DataFrame(q[1], columns=column_list_in)

        featured_data = du.generate_features(self.data_1min, 10, BUY_LIMIT, self.features, True)

        if featured_data is not None:
            if "Time" in featured_data.columns.to_list():
                now = datetime.datetime.now()
                minutes_since_open = (now.hour - MARKET_OPEN_HOUR)*60 + (now.minute + 1)
                featured_data["Time"] = featured_data["Time"] - (1 - minutes_since_open / (16 * 60))  # time within a day/16h

            self.last_predictions = self.make_predictions(featured_data)
            if len(self.last_predictions) >= 8:
                q = ("UPDATE", self.last_predictions + [self.process_rate])
                self.outQ.put_nowait(q)

    def on_data_end(self, q: Queue):
        self.on_data_update(q)

    def make_predictions(self, featured_data):
        if len(featured_data) == 0:
            print("Empty Input Data")
            logging.error("Empty Input Data")
            return []
        else:
            compound_data = pd.DataFrame()

            compound_data["Close"] = featured_data.pop("Close")
            compound_data["High"] = featured_data.pop("High")

            compound_data["Confidence"] = du.predict_buy(self.model, self.scaler, featured_data, isNeural=IS_NEURAL)

            if BUY_LIMIT > 0.0:
                _change = np.concatenate(
                    ([0.0], compound_data["High"].to_numpy()[1:] - compound_data["Close"].to_numpy()[:-1]))
                compound_data["Change"] = _change / compound_data["Close"].shift(1)
            else:
                compound_data["Change"] = ta.change(compound_data["Close"]) / compound_data["Close"].shift(1)

            compound_data["isBuy"] = featured_data.isBuy.astype(int)
            compound_data["toBuy"] = (compound_data["Confidence"] > CONF_LEVEL).astype(int)
            compound_data["randBuy"] = (np.random.rand(compound_data.shape[0]) > 0.5).astype(int)

            compound_data["truePositive"] = np.logical_and(compound_data[["isBuy"]].astype(bool),
                                                           compound_data[["toBuy"]].astype(bool)).astype(int)
            compound_data["trueNegative"] = np.logical_and(~compound_data[["isBuy"]].astype(bool),
                                                           ~compound_data[["toBuy"]].astype(bool)).astype(int)
            compound_data["falsePositive"] = np.logical_and(~compound_data[["isBuy"]].astype(bool),
                                                            compound_data[["toBuy"]].astype(bool)).astype(int)
            compound_data["falseNegative"] = np.logical_and(compound_data[["isBuy"]].astype(bool),
                                                            ~compound_data[["toBuy"]].astype(bool)).astype(int)

            compound_data["cum_tP"] = compound_data["truePositive"].cumsum(skipna=True)
            compound_data["cum_tN"] = compound_data["trueNegative"].cumsum(skipna=True)
            compound_data["cum_fP"] = compound_data["falsePositive"].cumsum(skipna=True)
            compound_data["cum_fN"] = compound_data["falseNegative"].cumsum(skipna=True)

            compound_data["precision_Pos"] = 100 * compound_data["cum_tP"] / (
                        compound_data["cum_tP"] + compound_data["cum_fP"])
            compound_data["recall_Pos"] = 100 * compound_data["cum_tP"] / (
                        compound_data["cum_tP"] + compound_data["cum_fN"])
            compound_data["precision_Neg"] = 100 * compound_data["cum_tN"] / (
                        compound_data["cum_tN"] + compound_data["cum_fN"])
            compound_data["recall_Neg"] = 100 * compound_data["cum_tN"] / (
                        compound_data["cum_tN"] + compound_data["cum_fP"])

            compound_data["cum_perfect"] = 100 * (compound_data["isBuy"].shift(1) * compound_data["Change"]).cumsum(
                skipna=True)
            compound_data["cum_predicted"] = 100 * (compound_data["toBuy"].shift(1) * compound_data["Change"]).cumsum(
                skipna=True)
            compound_data["cum_random"] = 100 * (compound_data["randBuy"].shift(1) * compound_data["Change"]).cumsum(
                skipna=True)

            compound_data["profit_efficiency"] = 100 * compound_data["cum_predicted"] / compound_data["cum_perfect"]
            #compound_data["profit_efficiency_rand"] = 100 * compound_data["cum_random"] / compound_data["cum_perfect"]

            total_precision_recall = np.nan_to_num(compound_data[["precision_Pos", "recall_Pos", "precision_Neg", "recall_Neg"]].to_numpy()[-1])
            efficiencies = np.nan_to_num(compound_data[["cum_predicted", "cum_perfect"]].to_numpy()[-1])

            return [compound_data["Confidence"].values[-2], compound_data["Confidence"].values[-1]] + list(
                total_precision_recall) + list(efficiencies)


def main():
    pass


if __name__ == "__main__":
    main()