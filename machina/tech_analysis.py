"""
Module for tech analysis functions

candle is of form: np.array([0time, 1open, 2close, 3low, 4high])

"""

import time
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

def low(ar):
    """Input array is OHLC candle"""
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    return ar[:, 2]

def high(ar):
    """Input array is OHLC candle"""
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    return ar[:, 1]

def lowest(ar, length=1):
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(ar.size, length)  # length check
    return ar[-_length:].min()

def highest(ar, length=1):
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(ar.size, length)  # length check
    return ar[-_length:].max()

def movingH(ar, length=0):  # moving highest value over length periods : if 0 for all
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    if length == 0:
        return np.maximum.accumulate(ar)
    else:
        res = np.zeros_like(ar)
        for i, _ in enumerate(res):
            _slice = ar[max(i + 1 - length, 0):i + 1]
            res[i] = _slice.max()

        return res

def movingL(ar, length=0):  # moving lowest value over length periods : if 0 for all
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    if length == 0:
        return np.minimum.accumulate(ar)
    else:
        res = np.zeros_like(ar)
        for i, _ in enumerate(res):
            _slice = ar[max(i + 1 - length, 0):i + 1]
            res[i] = _slice.min()

        return res

def movingSign(ar, length=1):
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(length if length > 0 else 1, ar.size)  # length check
    _s_ar = np.sign(ar)
    _sum = np.cumsum(_s_ar)
    _sum[_length:] = _sum[_length:] - _sum[:-_length]
    return np.concatenate((_s_ar[:_length - 1], _sum[_length - 1:] / _length))

def rowSign(ar):
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()

    ar = np.sign(ar)
    output = np.zeros_like(ar)
    for i, el in enumerate(ar):
        if i == 0:  # first element
            output[i] = el
        else:
            if el == ar[i-1]:  # the same sign
                output[i] = el + output[i-1]
            else:  # change of sign = reset
                output[i] = el

    return output

def change(ar):
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    #print(ar)
    return np.concatenate(([0.0], ar[1:] - ar[:-1]))

def isBuy(ar, buyLimit=0.0):   # buylimit in fraction of true range
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()

    _open = ar[:, 0]
    _high = ar[:, 1]
    _low = ar[:, 2]
    _close = ar[:, 3]
    _limit = tr(ar) * buyLimit

    if buyLimit > 0.0:
        out = np.concatenate((_high[1:] - _close[:-1], [0.0]))
    else:
        out = np.concatenate((_close[1:] - _close[:-1], [0.0]))

    out[out > _limit] = 1
    out[out <= _limit] = 0

    return out

def sma(ar, length=1):
    """
    The sma function returns the moving average, that is the sum of last "length" values of "a", divided by "length".
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(length if length > 0 else 1, ar.size)  # length check
    if _length == 1:
        return ar
    else:
        _sum = np.cumsum(ar)
        _sum[_length:] = _sum[_length:] - _sum[:-_length]
        _sma = sma(ar[:_length - 1], _length-1)
        return np.concatenate((_sma, _sum[_length - 1:] / _length))

def rma(ar, length=1):
    """
    Exponential moving average used in RSI.
    It is the exponentially weighted moving average with alpha = 1 / length.
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _n = ar.size
    _length = min(length if length > 0 else 1, _n)  # length check
    _alpha = 1 / _length
    _alpha_m1 = 1 - _alpha

    _rma = np.zeros(_n)
    _rma[0] = ar[0]
    for i in range(1, _n):
        _rma[i] = ar[i] * _alpha + _rma[i-1] * _alpha_m1

    return _rma

def ema(ar, length=1):
    """
    The ema function returns the exponentially weighted moving average.
    In ema weighting factors decrease exponentially.
    It calculates by using a formula: EMA = alpha * x + (1 - alpha) * EMA[1], where alpha = 2 / (length + 1)
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _n = ar.size
    _length = min(length if length > 0 else 1, _n)  # length check
    _alpha = 2 / (_length + 1)
    _alpha_m1 = 1 - _alpha

    _ema = np.zeros(_n)
    _ema[0] = ar[0]
    for i in range(1, _n):
        _ema[i] = ar[i] * _alpha + _ema[i-1] * _alpha_m1

    return _ema

def tr(ar):
    """
    True range
    First element is not fully valid
    Input array is OHLC candle
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _open = ar[:, 0]
    _high = ar[:, 1]
    _low = ar[:, 2]
    _close = ar[:, 3]

    _max = np.amax(
        np.transpose([_high[1:] - _low[1:], np.abs(_high[1:] - _close[:-1]), np.abs(_low[1:] - _close[:-1])]), axis=1)
    return np.abs(np.concatenate(([_high[0] - _low[0]], _max)))

def atr(ar, length=1):
    """
    Function atr (average true range) returns the RMA of true range.
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    return rma(tr(ar), length)

def vwap(ar):  # wap/volume
    """
    Function vwap: volume-weightned average price.
    """
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    if ar[0, 1] == 0:
        ar[0, 1] = 1e-2

    return np.cumsum(ar[:, 0] * ar[:, 1])/np.cumsum(ar[:, 1])


def cci(ar, length=1):  # high/low/close, length
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(length if length > 0 else 1, ar.size)  # length check

    high = ar[:, 0]
    low = ar[:, 1]
    close = ar[:, 2]

    t_price = (high + low + close) / 3
    move_avrg = sma(t_price, _length)
    mean_dev = sma(np.abs(t_price-move_avrg), _length)

    return np.divide((t_price-move_avrg), 0.015*mean_dev, out=np.zeros_like(high, dtype=float), where=mean_dev != 0)

def stoch(ar, length_pK=1, length_sD=1, length_sK=1):  # high/low/close, lengths
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length_pK = min(length_pK if length_pK > 0 else 1, ar.size)  # length check
    _length_sD = min(length_sD if length_sD > 0 else 1, ar.size)  # length check
    _length_sK = min(length_sK if length_sK > 0 else 1, ar.size)  # length check

    _high = ar[:, 0]
    _low = ar[:, 1]
    _close = ar[:, 2]

    _movL = movingL(_low, _length_pK)
    _movH = movingH(_high, _length_pK)
    _div = np.divide(100 * (_close - _movL), (_movH - _movL), out=50*np.ones_like(_high, dtype=float), where=_movL != _movH)
    _K = sma(_div, _length_sK)
    _D = sma(_K, _length_sD)

    return _K, _D

def rsi(ar, length=1):  # high/low/close, lengths
    if not isinstance(ar, np.ndarray): ar = ar.to_numpy()
    _length = min(length if length > 0 else 1, ar.size)  # length check

    _chng = change(ar)
    _up = np.maximum(_chng, 0)  # upward
    _down = np.maximum(-_chng, 0)  # downward
    rma_up = rma(_up, _length)
    rma_down = rma(_down, _length)
    rs = np.divide(rma_up, rma_down, out=np.ones_like(ar, dtype=float), where=rma_down != 0)

    return 100 - 100 / (1 + rs)

def main():
    now = time.time()
    t = np.linspace(now - 6 * 30 * 24 * 3600, now, 200)  # Plot random values with timestamps in the last 6 months
    slowC = np.transpose([t,
                          15 + 3 * np.abs(np.sin(t)),
                          15 + 3 * np.abs(np.sin(t + 1.5)),
                          15 - 6 * np.abs(np.sin(t)),
                          15 + 6 * np.abs(np.sin(t))])

    _tr = tr(slowC[:, 1:])
    _atr = atr(slowC[:, 1:], 10)
    _sma = sma(slowC[:, 1], 10)
    print(f"{slowC.shape}/{_sma.shape}")
    t1 = time.perf_counter()
    # _ema = ema(slowC[:, 1], 100)
    _rma = rma(slowC[:, 1], 10)
    _cci = cci(slowC[:, 2:], 10)
    _rsi = rsi(slowC[:, 4], 5)
    _stochK, _stochD = stoch(slowC[:, 1:], 5, 3, 3)
    t2 = time.perf_counter()
    #print(_tr)

    fig, ax = plt.subplots()
    ax.plot(slowC[:, 0], _cci)
    #ax.plot(slowC[:, 0], _stochD)
    plt.show()


if __name__ == "__main__":
    main()
