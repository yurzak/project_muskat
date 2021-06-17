import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import tech_analysis as ta

# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.experimental import enable_hist_gradient_boosting
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
#
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import metrics

def predict_buy(model, scaler, data_in, isNeural=True):
    scaled = scaler.transform(data_in.iloc[:, :-1])  # i.e. without last column/target
    x = np.array(scaled,)
    if isNeural:
        predict = 100 * model.predict(x)  # confidence in percents
    else:
        predict = 100 * model.predict_proba(x)[:, 1]  # confidence in percents

    return predict

def use_record(title: str, date: list, price: list, price_gain: list, volume, volume_gain: list) -> bool:
    """Return True/False based on filter conditions"""
    title = title.replace(".txt", "")
    title = title.split("_")
    title.pop(1)

    markers = ["H", "V", "G", "VF"]
    for i, marker in enumerate(markers):
        title[i+1] = int(title[i + 1].replace(marker, ""))
    title[0] = int(title[0])

    if title[0] < date[0] or title[0] > date[1]:
        return False
    elif title[1] < price[0] or title[1] > price[1]:
        return False
    elif title[2] < volume[0] or title[1] > volume[1]:
        return False
    elif title[3] < price_gain[0] or title[1] > price_gain[1]:
        return False
    elif title[4] < volume_gain[0] or title[1] > volume_gain[1]:
        return False
    else:
        return True


def generate_features(data_in, min_seq_length=1, buyLimit=0.0, features=None, isEval=False):
    out = pd.DataFrame()
    data_ohlc = data_in[["Open", "High", "Low", "Close"]]
    tr = ta.tr(data_ohlc)

    #valid_idx = (tr != 0)
    #print(tr)
    #print(data_ohlc.describe())

    if min_seq_length > 1:
        mask_ind = np.hstack([[[0]], np.where(tr == 0)])
        if mask_ind[0, -1] < data_ohlc.shape[0]-1:
            mask_ind = np.hstack([mask_ind, [[data_ohlc.shape[0]-1]]])

        #print("mask_ind", mask_ind)

        seq_lengths = mask_ind[0, 1:] - mask_ind[0, :-1]
        #print("seq_lengths", seq_lengths)

        valid_seq_start = np.array(np.where(seq_lengths > min_seq_length))  # still wil exclude tr==0 even for min_length=0
        #print("valid_seq_start", valid_seq_start)

        s_ind = mask_ind[0, valid_seq_start] + 1
        e_ind = mask_ind[0, valid_seq_start + 1]
        seq_limits = zip(s_ind[0, :], e_ind[0, :])  # inclusive limits
        #print("start/stop indices", s_ind, e_ind)

        valid_idx = []
        for start, stop in seq_limits:
            valid_idx += list(range(start, stop+1))

        data_in = data_in.iloc[valid_idx, :].reset_index()

        if len(valid_idx) == 0:  # no sequences with valid min length
            return None

        tr = tr[valid_idx]

    atr10 = ta.rma(tr, 10)
    zeros = np.zeros_like(atr10, dtype=float)

    out["Open_change"] = np.divide(ta.change(data_in["Open"]), atr10)  # open_change/atr10
    out["High_change"] = np.divide(ta.change(data_in["High"]), atr10)  # high_change/atr10
    out["Low_change"] = np.divide(ta.change(data_in["Low"]), atr10)  # low_change/atr10
    out["Close_change"] = np.divide(ta.change(data_in["Close"]), atr10)  # close_change/atr10

    out["Last_row"] = ta.rowSign(out["Close_change"])  # number-last green/red in a row
    out["Last5_ratio"] = ta.movingSign(out["Close_change"], 5)  # number_last5_green/red imbalance fraction
    out["Last10_ratio"] = ta.movingSign(out["Close_change"], 10)  # number_last10_green/red imbalance fraction

    _dist = np.divide((data_in["Close"] - data_in["Open"].iloc[0]), atr10)
    out["OpenDay_dist"] = np.sign(_dist)*np.log10(1 + np.abs(_dist))  # log10 of openDay_dist/atr10
    _dist = np.divide((ta.movingH(data_in["High"]) - data_in["Close"]), atr10)
    out["HighDay_dist"] = np.sign(_dist)*np.log10(1 + np.abs(_dist))  # log10 of highDay_dist/atr10
    _dist = np.divide((data_in["Close"] - ta.movingL(data_in["Low"])), atr10)
    out["LowDay_dist"] = np.sign(_dist)*np.log10(1 + np.abs(_dist))  # log10 of lowDay_dist/atr10
    _dist = np.divide((data_in["Close"] - ta.vwap(np.transpose([data_in["Wap"].to_numpy(), data_in["Volume"].to_numpy()]))), atr10)
    out["Vwap_dist"] = np.sign(_dist)*np.log10(1 + np.abs(_dist))  # log10 of vwap_dist/atr10

    out["Ema10_dist"] = np.divide((data_in["Close"] - ta.ema(data_in["Close"], 10)), atr10)  # ema10_dist/atr10
    out["Ema50_dist"] = np.divide((data_in["Close"] - ta.ema(data_in["Close"], 50)), atr10)  # ema50 dist/atr10

    out["Ema10_change"] = np.divide(ta.ema(ta.change(ta.ema(data_in["Close"], 10)), 10), atr10)   # ema10 of ema10_change/atr10
    out["Ema50_change"] = np.divide(ta.ema(ta.change(ta.ema(data_in["Close"], 50)), 10), atr10)  # ema10 of ema50_change/atr10

    ma = ta.sma(data_in["Volume"], 2)
    out["Volume_change"] = np.divide(ta.change(data_in["Volume"]), ma, out=zeros, where=ma != 0)  # volume_change_per
    ma = ta.sma(data_in["Count"], 2)
    out["Trades_change"] = np.divide(ta.change(data_in["Count"]), ma, out=zeros, where=ma != 0)  # trades_change_per
    ma = ta.ema(data_in["Volume"], 10)
    out["VolumeEma10_dist"] = np.divide(data_in["Volume"], ma, out=zeros, where=ma != 0) - 1  # volume_ema10_dist

    out["CCI"] = ta.cci(data_in[["High", "Low", "Close"]], 20)
    out["CCI_ema"] = ta.ema(out["CCI"], 10)
    out["CCI_diff"] = out["CCI"]-out["CCI_ema"]

    out["Stoch_fast"], out["Stoch_slow"] = ta.stoch(data_in[["High", "Low", "Close"]], 5, 3, 3)
    out["Stoch_diff"] = out["Stoch_fast"] - out["Stoch_slow"]

    out["RSI"] = ta.rsi(data_in["Close"], 5)
    out["RSI_ema"] = ta.ema(out["RSI"], 10)
    out["RSI_diff"] = out["RSI"] - out["RSI_ema"]

    out["RSI_vol"] = ta.rsi(np.sign(out["Close_change"].to_numpy())*data_in["Volume"].to_numpy(), 5)
    out["RSI_vol_ema"] = ta.ema(out["RSI_vol"], 10)
    out["RSI_vol_diff"] = out["RSI_vol"] - out["RSI_vol_ema"]

    out["ATR_norm"] = np.divide(atr10, ta.movingH(atr10))  # atr10/atrMax
    out["Tr"] = np.divide(tr, atr10)  # tr/atr10
    out["Time"] = (data_in["Time"] - data_in["Time"].iloc[-1]) / (16 * 60 * 60) + 1  # time within a day/16h

    out["isBuy"] = ta.isBuy(data_in[["Open", "High", "Low", "Close"]], buyLimit)  # buy/sell aka next 1min_change_sign or fraction of tr

    if isEval:
        out["Close"] = data_in["Close"]
        out["High"] = data_in["High"]
        return out[features+["Close", "High"]]

    return out

def evaluate_company_day(model, scaler, path, min_seq_length, buyLimit, features, confidence_level, isNeural=True):
    column_list_in = ["Time", "Open", "High", "Low", "Close", "Volume", "Count", "Wap"]
    frame_in = pd.read_csv(path, usecols=column_list_in)
    featured_data = generate_features(frame_in, min_seq_length=min_seq_length, buyLimit=buyLimit,
                                      features=features, isEval=True)
    if featured_data is None:
        return
    elif len(featured_data) == 0:
        return None
    else:
        compound_data = pd.DataFrame()

        compound_data["Close"] = featured_data.pop("Close")
        compound_data["High"] = featured_data.pop("High")

        if buyLimit > 0.0:
            _change = np.concatenate(([0.0], compound_data["High"].to_numpy()[1:] - compound_data["Close"].to_numpy()[:-1]))
            compound_data["Change"] = _change / compound_data["Close"].shift(1)
        else:
            compound_data["Change"] = ta.change(compound_data["Close"]) / compound_data["Close"].shift(1)

        compound_data["isBuy"] = featured_data.isBuy.astype(int)
        compound_data["toBuy"] = (predict_buy(model, scaler, featured_data, isNeural=isNeural) > confidence_level).astype(int)
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

        compound_data["precision_Pos"] = 100 * compound_data["cum_tP"] / (compound_data["cum_tP"] + compound_data["cum_fP"])
        compound_data["recall_Pos"] = 100 * compound_data["cum_tP"] / (compound_data["cum_tP"] + compound_data["cum_fN"])
        compound_data["precision_Neg"] = 100 * compound_data["cum_tN"] / (compound_data["cum_tN"] + compound_data["cum_fN"])
        compound_data["recall_Neg"] = 100 * compound_data["cum_tN"] / (compound_data["cum_tN"] + compound_data["cum_fP"])

        compound_data["cum_perfect"] = 100 * (compound_data["isBuy"].shift(1) * compound_data["Change"]).cumsum(skipna=True)
        compound_data["cum_predicted"] = 100 * (compound_data["toBuy"].shift(1) * compound_data["Change"]).cumsum(skipna=True)
        compound_data["cum_random"] = 100 * (compound_data["randBuy"].shift(1) * compound_data["Change"]).cumsum(skipna=True)

        compound_data["profit_efficiency"] = 100 * compound_data["cum_predicted"] / compound_data["cum_perfect"]
        compound_data["profit_efficiency_rand"] = 100 * compound_data["cum_random"] / compound_data["cum_perfect"]

        total_precision_recall = np.nan_to_num(compound_data[["precision_Pos", "recall_Pos", "precision_Neg", "recall_Neg"]].values[-1])
        mean_precision_recall = compound_data[["precision_Pos", "recall_Pos", "precision_Neg", "recall_Neg"]].mean(skipna=True).values
        efficiencies = np.nan_to_num(compound_data[["profit_efficiency", "profit_efficiency_rand", "cum_predicted", "cum_random", "cum_perfect"]].values[-1])

        return list(total_precision_recall) + list(mean_precision_recall) + list(efficiencies)

def main():
    print("starting")

    # cwd = os.getcwd()
    # #path = cwd + "\\data\\1min\\nasdaq\\20200601_DGLY_H3_V93_G149_VF12.txt"
    # path = cwd + "\\data\\1min\\nyse\\20200110_SNX_H74_V4_G15_VF3.txt"
    #
    # compound = evaluate_company_day(MODEL, SCALER, path, [-2e2, 2e2], 50)
    #
    # compound.tail(20)
    # plt.figure(figsize=(10,5))
    # #featured_data.loc[:,"Ema10_dist"].plot()
    # compound_data["Close"].plot()
    # compound[["precision_Pos"]].plot()
    #
    cwd = os.getcwd()
    path = cwd + "\\data\\1min\\nasdaq\\20200601_DGLY_H3_V93_G149_VF12.txt"#nyse\\20200110_SNX_H74_V4_G15_VF3.txt"#nasdaq\\20200601_DGLY_H3_V93_G149_VF12.txt"
    column_list_in = ["Time", "Open", "High", "Low", "Close", "Volume", "Count", "Wap"]
    frame_in = pd.read_csv(path, usecols=column_list_in)
    featured_data = generate_features(frame_in, min_seq_length=0)

    plt.figure(figsize=(20, 5))
    # frame_in.loc[:,"Close"].plot()
    # featured_data.loc[540:580,"isBuy"].plot()
    # featured_data.loc[540:580,"Last_row"].plot()
    #featured_data.loc[:, "Vwap_dist"].plot()
    # featured_data.Close.plot()
    # featured_data.Ema10.plot()
    # featured_data.Ema50.plot()
    featured_data.Ema50_dist.plot()
    #featured_data.Ema10_changeEma10.plot()
    #featured_data.ATR_norm.plot()
    #print(featured_data.describe())
    plt.show()

if __name__ == "__main__":
    main()