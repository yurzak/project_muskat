# -*- coding: utf-8 -*-
"""
Module for Interactive Brokers Application, Wrapper/Client methods
"""

import time as tm
import logging

import numpy as np

from ibapi import (wrapper, decoder, reader, comm, utils)
from ibapi.client import EClient
from ibapi.utils import (current_fn_name, BadMessage)
from ibapi.execution import Execution
from ibapi.scanner import ScanData
from ibapi.scanner import ScannerSubscription
from ibapi.tag_value import TagValue

from ibapi.order_condition import * # @UnusedWildImport
from ibapi.contract import * # @UnusedWildImport
from ibapi.order import * # @UnusedWildImport
from ibapi.order_state import * # @UnusedWildImport
from ibapi.common import * # @UnusedWildImport
from ibapi.ticktype import * # @UnusedWildImport
from ibapi.errors import * #@UnusedWildImport

from multiprocessing import Queue
import queue

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7497

IB_QUEUE_RATE = 0.005  # reading time of queues in seconds
IB_SERVER_RATE = 0.01  # reading time of queues in seconds

OUTSIDE_RTH = True  # default value for limit order execution outside regular trading hours
XSLOW_SPAN = "1 Y"  # S D W M Y  # extra-slow history candles time span (for day-bars)
SLOW_SPAN = "2 D"  # S D W M Y  # slow history candles time span
FAST_SPAN = "1800 S"  # S D W M Y  # fast history candles time span
HIGHLOW_SPAN = "1000 S"  # S D W M Y  # fast history candles time span

SCAN_PRICE_MIN = 1
SCAN_PRICE_MAX = 15

TICKS_HISTORY_NUMBER = 1000  # max = 1000
LEVEL2_QUEUE_RATE = 10  # defines max rate for Level2 bid/ask/trades/order book updates (not to block queue with on too frequent updates)
LEVEL2_OB_BUFFER_SIZE = 10*1000  # typically 1000 records per 100 ms

NEWS_NUMBER = 5  # number of news titles
MAX_SCAN_DURATION = 3  # in seconds

class IbApp(wrapper.EWrapper, EClient):
    """compound EWrapper and EClient class=> IB_API"""
    def __init__(self, inQ:Queue, outQ:Queue, outL1Q:Queue, outL2Q:Queue, outMLQ:Queue):
        wrapper.EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.inQ = inQ  # from gui
        self.outQ = outQ  # to gui 1
        self.outL1Q = outL1Q  # to gui2
        self.outL2Q = outL2Q  # to gui 3
        self.outMLQ = outMLQ  # to ml
        self.connectReq = False
        self.connectionOK = False
        self.exitReq = False
        self.ibAPI_time = tm.perf_counter()
        #  ----------local values to be sent via queue
        self.isAccPortStreaming = False

        self.NetLiquidationByCurrency = 0.0  # total account value = cash+stocks
        self.StockMarketValue = 0.0  # stocks value
        self.TotalCashBalance = 0.0  # free cash available
        self.RealizedPnL = 0.0  # stocks realized PnL (total of a day?)
        self.UnrealizedPnL = 0.0  # stocks unrealized PnL (total of a day?)

        self.posSize = 0  # number of shares
        self.posPrice = 0.0  # current price per share
        self.posCost = 0.0  # average paid per share
        self.posRealizedPnL = 0.0  # realized total PnL
        self.posUnrealizedPnL = 0.0  # unrealized total PnL
        #  --------------------------------------------
        self.contract = Contract()
        self.contract.symbol = "TSLA"
        #self.contract.localSymbol = "TSLA"
        self.contract.secType = "STK"
        self.contract.currency = "USD"
        self.contract.exchange = "SMART"
        #self.contract.PrimaryExch = "NASDAQ"

        self.newscontract = Contract()
        self.newscontract.symbol = "TSLA"
        #self.newscontract.localSymbol = "TSLA"
        self.newscontract.secType = "STK"
        self.newscontract.currency = "USD"
        self.newscontract.exchange = "SMART"
        #self.newscontract.PrimaryExch = "NASDAQ"

        self.validExchanges = ""
        self.longName = ""
        self.newscontractId = None  # news company ID
        self.priceMinIncrement = 0.01
        self.priceDecimals = 2
        #  --------order
        self.isOrderCompleted = True
        self.nextValidOrderId = None
        self.activeOrderId = None
        self.positionData = np.array([])
        #  ----------DATA Processing/streaming
        self.contractId = 0  # contract/ticker details
        self.isSlowHistoryStreaming = False
        self.slowHistoryId = 1  # TRADES/candles history
        self.HighLowHistoryId = 2  # TRADES/candles history for high/low values
        self.isFastHistoryStreaming = False
        self.fastHistoryId = 3  # TRADES/candles history
        self.fastHistoryId_bid = 4  # for BID history updates
        self.fastHistoryId_ask = 5  # for ASK history updates
        self.isLevel1Streaming = False
        self.streamLevel1Id = 6
        self.streamLastTickId = 7
        self.streamBidAskTickId = 8
        self.isLevel2Streaming = False
        self.streamLevel2Id = 9
        self.streamNYSE = True

        self.newsId = 10  # NEWS request
        self.isScanning = False
        self.isScanDataComplete = False
        self.scannerId = 11
        self.scannerDataId = 100
        self.last_Scan_time = tm.time()

        self.timeLevel2TradesUpdate = tm.perf_counter()
        self.timeLevel2BidAskUpdate = tm.perf_counter()
        self.timeOrderBookUpdate = tm.perf_counter()

        self.newsData = []
        self.scanList = np.array([])
        self.scanData = np.array([])
        #  ---DATA ARRAYS---------------
        self.data_slowCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
        self.data_1min = np.array([[]])  # history + 5s? updates: time, OHLC, volume,trades, vwap
        self.data_1min_HighLows = np.array([[]])  # history + 5s? updates: time, max, min
        self.data_slowBars = np.array([])  # history + 5s? updates: time, volume
        self.data_slowCurve = np.array([])  # history + 5s? updates: time, number of trades
        self.data_fastCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
        self.data_fastBars = np.array([])  # history + 5s? updates: time, volume
        self.data_fastCurve = np.array([])  # history + 5s? updates: time, number of trades
        self.data_fastCurveBid = np.array([])  # history + 5s? updates: time, bid
        self.data_fastCurveAsk = np.array([])  # history + 5s? updates: time, bid
        self.data_tick = np.array([])  # 250ms MarketData: bid, ask, bidvolume, ask volume, last, last V, etc...

        self.data_level2Scatter_ticks = []#np.array([])  # [time, price, size]
        self.data_level2BACurves_ticks = []#np.array([])  # TickTick BA: time, bidPrice, askPrice, bidSize, askSize
        self.data_level2OB_cache = np.zeros((LEVEL2_OB_BUFFER_SIZE, 5), float)  #np.array([[]])
        self.level2OB_cache_index = 0 # current index to write within the buffer
        self.data_orderBookBidBars = np.zeros((1, 2), float)  # order book_bid: price, size
        self.data_orderBookAskBars = np.zeros((1, 2), float)  # order book_ask: price, size

    def readQ(self):
        """Read Queue (inQ) method and call corresponding handler method"""
        try:
            #q = self.inQ.get(timeout=IB_QUEUE_RATE)
            q = self.inQ.get_nowait()
        except queue.Empty:
            pass  # q = ("",)
        else:
            request = q[0]
            if request == "CONNECT":
                self.req_Connect(q)
            if request == "ACC_PORT":
                self.req_AccountPortfolio()
            if request == "ACC_PORT_STOP":
                self.req_AccountPortfolioStop()
            if request == "FIND":
                self.req_Find(q)
            if request == "NEWS":
                self.req_GetNews(q)
            if request == "SCAN":
                self.req_GetScan(q)
            if request == "LEVEL1":
                self.req_Level1(q)
            if request == "LEVEL1_STOP":
                self.req_Level1Stop()
            if request == "LEVEL2":
                self.req_Level2(q)
            if request == "LEVEL2_STOP":
                self.req_Level2Stop()
            if request == "LEVEL2_NYSE":
                self.streamNYSE = q[1]
            if request == "PLACE_ORDER":
                self.req_PlaceOrder(q)
            if request == "CANCEL_ORDER":
                self.req_CancelOrder(q)
            if request == "EXIT":
                self.req_Exit(q)

    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Overridden EWrapper: errors received"""
        #super().error(reqId, errorCode, errorString)

        if 501 <= errorCode <= 504 or 1100 <= errorCode <= 1102 or errorCode == 1300:
            q = ("CONNECTION_ERROR", errorCode)
            self.outQ.put_nowait(q)
            logging.error(f"Connection Error: {errorCode}")
            #print("Connection Error")
        elif errorCode == 2100:
            q = ("ACC_PORT_UPDATE_STOPPED",)
            self.outQ.put_nowait(q)
            logging.info("Account/Portfolio updates stopped")
        elif errorCode == 162:
            logging.debug("API historical data query cancelled")
        elif errorCode == 366:
            logging.debug("No historical data query found for ticker (cancel requested, but already cancelled?)")
        else:
            if errorCode == 200 and self.isScanning:
                #print('Ambiguous ticker')
                for ind, ticker in enumerate(self.scanList):  # delete ticker from the list=replace with ""
                    if ticker in errorString:
                        self.scanList[ind] = "*"
            else:
                # if errorCode == 202:
                #     #print('Order canceled')
                q = ("ERROR", errorCode, errorString)
                self.outQ.put_nowait(q)
                logging.error(f"Error: {errorCode}, {errorString}")

        #print("Error. Id:", reqId, "Code:", errorCode, "Msg:", errorString)

    def headTimestamp(self, reqId:int, headTimestamp:str):
        print("HeadTimestamp. ReqId:", reqId, "HeadTimeStamp:", headTimestamp)

    def req_Connect(self, q: Queue):
        """Connection requested"""
        if not self.connectReq:
            self.connectReq = True
            #print("Connecting to TWS...")
            self.connect(host=q[1], port=q[2], clientId=q[3])
            # self.reqAccountUpdates(True, "DU")
            logging.info("Connecting to TWS")

    def req_AccountPortfolio(self):
        """Account/Portfolio Updates requested"""
        if not self.isAccPortStreaming:
            #print("Account/Portfolio/Positions Update starting...")
            self.isAccPortStreaming = True
            logging.info("Account/Portfolio updates requested")
            self.reqAccountUpdates(True, "")

    def req_AccountPortfolioStop(self):
        """Account/Portfolio Updates Stop requested"""
        if self.isAccPortStreaming:
            #print("Account/Portfolio Update stoping...")
            self.isAccPortStreaming = False
            logging.info("Account/Portfolio updates stop requested")
            self.reqAccountUpdates(False, "")

    def req_Find(self, q: Queue):
        """Find company requested"""
        self.contract.symbol = q[1]
        self.reqContractDetails(self.contractId, self.contract)
        logging.info("Contract/company details requested")
        if not self.isSlowHistoryStreaming:
            #print("Slow History+no updates starting...")
            self.cancelHistoricalData(self.slowHistoryId)
            self.cancelHistoricalData(self.HighLowHistoryId)
            # clear values for next run
            self.data_slowCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
            self.data_1min = np.array([[]])  # history + 5s? updates: time, OHLC, volume,trades, vwap
            self.data_1min_HighLows = np.array([[]])  # history + 5s? updates: time, max, min
            self.data_slowBars = np.array([])  # history + 5s? updates: time, volume
            self.data_slowCurve = np.array([])  # history + 5s? updates: time, number of trades
            _span = XSLOW_SPAN if q[2] == "1 day" else FAST_SPAN if "sec" in q[2] else SLOW_SPAN
            self.reqHistoricalData(self.slowHistoryId, self.contract, "", _span, q[2],
                                   "TRADES", q[4], 2, False, [])  # slow data
            self.reqHistoricalData(self.HighLowHistoryId, self.contract, "", _span, "1 min",
                                   "TRADES", q[4], 2, False, [])  # slow data
            logging.info("Slow history bars+no updates requested")
        if not self.isFastHistoryStreaming:
            #print("Fast History+no updates starting...")
            self.cancelHistoricalData(self.fastHistoryId)
            self.cancelHistoricalData(self.fastHistoryId_bid)
            self.cancelHistoricalData(self.fastHistoryId_ask)
            # clear values for next run
            self.data_fastCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
            self.data_fastBars = np.array([])  # history + 5s? updates: time, volume
            self.data_fastCurve = np.array([])  # history + 5s? updates: time, number of trades
            self.data_fastCurveBid = np.array([])  # history + 5s? updates: time, bid
            self.data_fastCurveAsk = np.array([])  # history + 5s? updates: time, ask

            self.reqHistoricalData(self.fastHistoryId, self.contract,
                                   "", FAST_SPAN, q[3], "TRADES", q[4], 2, False, [])  # fast data
            self.reqHistoricalData(self.fastHistoryId_bid, self.contract,
                                   "", FAST_SPAN, q[3], "BID", q[4], 2, False, [])  # fast data bid
            self.reqHistoricalData(self.fastHistoryId_ask, self.contract,
                                   "", FAST_SPAN, q[3], "ASK", q[4], 2, False, [])  # fast data ask

            logging.info("Fast history bars+no updates requested")

        #self.reqHeadTimeStamp(150, self.contract, "TRADES", 1, 1)

    def req_GetNews(self, q: Queue):
        """Level1 stream requested=History+5s updates+MarketData"""
        logging.info("Get News requested")
        #self.reqNewsProviders()
        self.newscontract.symbol = q[1]
        self.reqContractDetails(self.newsId, self.newscontract)

    def req_GetScan(self, q: Queue):
        """Level1 stream requested=History+5s updates+MarketData"""
        _mode = q[1]
        _location = q[2]
        logging.info("Scanner Start requested")
        if not self.isScanning:
            #print("Scanner starting...")
            # self.reqScannerParameters()

            # Filters
            _tagvalues = []
            _tagvalues.append(TagValue("sharesOutstandingBelow", "200000000"))
            #_tagvalues.append(TagValue("sharesAvailableManyBelow", "50000000")) aka shorts
            _tagvalues.append(TagValue("priceAbove", str(SCAN_PRICE_MIN)))
            _tagvalues.append(TagValue("priceBelow", str(SCAN_PRICE_MAX)))
            _scanSub = ScannerSubscription()
            _scanSub.instrument = "STK"

            if _location == "MA":
                _scanSub.locationCode = "STK.US.MAJOR"
            elif _location == "NDQ":
                _scanSub.locationCode = "STK.NASDAQ"
            elif _location == "SCM":
                _scanSub.locationCode = "STK.NASDAQ.SCM"
            elif _location == "NMS":
                _scanSub.locationCode = "STK.NASDAQ.NMS"
            elif _location == "NY":
                _scanSub.locationCode = "STK.NYSE"
            elif _location == "AMX":
                _scanSub.locationCode = "STK.AMEX"
            elif _location == "ARC":
                _scanSub.locationCode = "STK.ARCA"
            elif _location == "BTS":
                _scanSub.locationCode = "STK.BATS"
            elif _location == "US":
                _scanSub.locationCode = "STK.US"

            if _mode == "3mVol":
                _scanSub.scanCode = "HIGH_STVOLUME_3MIN"
            elif _mode == "5mVol":
                _scanSub.scanCode = "HIGH_STVOLUME_5MIN"
            elif _mode == "10mVol":
                _scanSub.scanCode = "HIGH_STVOLUME_10MIN"
            elif _mode == "Gain":
                _scanSub.scanCode = "TOP_PERC_GAIN"
            elif _mode == "Lose":
                _scanSub.scanCode = "TOP_PERC_LOSE"
            elif _mode == "Gap+":
                _scanSub.scanCode = "HIGH_OPEN_GAP"
            elif _mode == "Gap-":
                _scanSub.scanCode = "LOW_OPEN_GAP"
            elif _mode == "Perf+":
                _scanSub.scanCode = "TOP_OPEN_PERC_GAIN"
            elif _mode == "Perf-":
                _scanSub.scanCode = "TOP_OPEN_PERC_LOSE"
            elif _mode == "AfterH+":
                _scanSub.scanCode = "TOP_AFTER_HOURS_PERC_GAIN"
            elif _mode == "AfterH-":
                _scanSub.scanCode = "TOP_AFTER_HOURS_PERC_LOSE"
            elif _mode == "Halted":
                _scanSub.scanCode = "HALTED"
            elif _mode == "HotPrice":
                _scanSub.scanCode = "HOT_BY_PRICE"
            elif _mode == "HotVol":
                _scanSub.scanCode = "HOT_BY_VOLUME"

            self.reqScannerSubscription(self.scannerId, _scanSub, [], _tagvalues)
            self.isScanning = True
        else:
            self.isScanning = False
            for ind, ticker in enumerate(self.scanList):  # cancel market data for every scan ticker
                if ticker != "*":
                    self.cancelMktData(self.scannerDataId + ind)
        # clear values for next run
        self.scanList = np.array([])
        self.scanData = np.zeros((0, 9), float)  # last/volume/previous day/bid/ask
        self.isScanDataComplete = False

    def req_Level1(self, q: Queue):
        """Level1 stream requested=History+5s updates+MarketData"""
        self.contract.symbol = q[1]
        self.reqContractDetails(self.contractId, self.contract)
        logging.info("Contract/company details requested")
        logging.info("History and Level1 requested")
        if not self.isLevel1Streaming:
            #print("History and Level1 starting...")
            # clear values for next run
            self.data_slowCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
            self.data_1min = np.array([[]])  # history + 5s? updates: time, OHLC, volume,trades, vwap
            self.data_1min_HighLows = np.array([[]])  # history + 5s? updates: time, max, min
            self.data_slowBars = np.array([])  # history + 5s? updates: time, volume
            self.data_slowCurve = np.array([])  # history + 5s? updates: time, number of trades
            self.data_fastCandles = np.array([])  # history + 5s? updates: time, open, close, min, max
            self.data_fastBars = np.array([])  # history + 5s? updates: time, volume
            self.data_fastCurve = np.array([])  # history + 5s? updates: time, number of trades
            self.data_fastCurveBid = np.array([])  # history + 5s? updates: time, bid
            self.data_fastCurveAsk = np.array([])  # history + 5s? updates: time, ask
            # has 5 secs min-step
            _span = XSLOW_SPAN if q[2] == "1 day" else FAST_SPAN if "sec" in q[2] else SLOW_SPAN
            self.reqHistoricalData(self.slowHistoryId, self.contract,
                                   "", _span, q[2], "TRADES", q[4], 2, True, [])  # slow data
            self.reqHistoricalData(self.HighLowHistoryId, self.contract,
                                   "", "1 D", "1 min", "TRADES", q[4], 2, True, [])  # 1 min high low data
            self.reqHistoricalData(self.fastHistoryId, self.contract,
                                   "", FAST_SPAN, q[3], "TRADES", q[4], 2, True, [])  # fast data
            self.reqHistoricalData(self.fastHistoryId_bid, self.contract,
                                   "", FAST_SPAN, q[3], "BID", q[4], 2, True, [])  # fast data BID
            self.reqHistoricalData(self.fastHistoryId_ask, self.contract,
                                   "", FAST_SPAN, q[3], "ASK", q[4], 2, True, [])  # fast data ASK
            # has 250 msecs min-step. here different values can be requested: bid/ask/size/las/volume/etc
            self.reqMktData(self.streamLevel1Id, self.contract, "165, 295", False, False, [])
            self.isLevel1Streaming = True

            self.outMLQ.put_nowait(("OPEN_MODEL",))  # opem MLCrode model

    def req_Level1Stop(self):
        """Level1 stream Stop requested"""
        logging.info("History and Level1 updates stop requested")
        if self.isLevel1Streaming:
            #print("History and Level1 stopping...")
            self.isLevel1Streaming = False
            self.cancelHistoricalData(self.slowHistoryId)
            self.cancelHistoricalData(self.HighLowHistoryId)
            self.cancelHistoricalData(self.fastHistoryId)
            self.cancelHistoricalData(self.fastHistoryId_bid)
            self.cancelHistoricalData(self.fastHistoryId_ask)
            self.cancelMktData(self.streamLevel1Id)

    def req_Level2(self, q: Queue):
        """Level2 stream requested=MarketDepth+TickTick"""
        self.contract.symbol = q[1]
        logging.info("Stream Tick-By-Tick data and Level2 requested")
        if not self.isLevel2Streaming:
            #print("History and Level2 starting...")
            # clear the data
            self.data_level2Scatter = np.array([])  # TickTick Last trades: time, price, size
            self.data_level2Scatter_ticks = []#np.array([])  # [time, price, size]
            self.data_level2BACurves_ticks = []#np.array([])  # time, bid, ask, bisize, asksize: later two are used for analytical
            self.data_volumeProfBars = np.array([])  # [price, Up volume, down volume]
            self.data_level2OB_cache = np.zeros((LEVEL2_OB_BUFFER_SIZE, 5), float)#np.array([[]])  # cahche for order book data
            self.data_orderBookBidBars = np.zeros((int(q[2]), 2), float)  # order book_bid: price, size
            self.data_orderBookAskBars = np.zeros((int(q[2]), 2), float)  # order book_ask: price, size
            #  all real-time/tick-based: 3 channels occupied out of 3 allowed
            self.reqTickByTickData(self.streamBidAskTickId, self.contract, "BidAsk", TICKS_HISTORY_NUMBER, False)
            self.reqTickByTickData(self.streamLastTickId, self.contract, "AllLast", TICKS_HISTORY_NUMBER, False)
            self.reqMktDepth(self.streamLevel2Id, self.contract, int(q[2]), True, [])
            self.isLevel2Streaming = True

    def req_Level2Stop(self):
        """Level2 stream Stop requested"""
        logging.info("Stream Tick-By-Tick data and Level2 stop requested")
        if self.isLevel2Streaming:
            #print("History and Level2 stopping...")
            self.isLevel2Streaming = False
            self.cancelTickByTickData(self.streamBidAskTickId)
            self.cancelTickByTickData(self.streamLastTickId)
            self.cancelMktDepth(self.streamLevel2Id, True)

    def req_PlaceOrder(self, q: Queue):
        """Order Buy/Sell requested"""
        self.contract.symbol = q[1]
        logging.info("Place Order requested")

        if self.connectionOK:
            self.activeOrderId = self.nextValidOrderId
            _type = q[2]
            _action = str(q[3])
            _size = float(q[4])

            if _type == "LMT":
                _price = float(q[5])
                _order = self.LimitOrder(_action, _size, _price)  # action, size, price
                self.placeOrder(self.nextValidOrderId, self.contract, _order)
            elif _type == "MKT":
                _order = self.MarketOrder(_action, _size)  # action, size
                self.placeOrder(self.nextValidOrderId, self.contract, _order)
            elif _type == "MP":
                _order = self.MidpriceOrder(_action, _size)  # action, size, price cap:optional
                self.placeOrder(self.nextValidOrderId, self.contract, _order)
            elif _type == "STP":
                _price = float(q[5])
                _order = self.StopOrder(_action, _size, _price)  # action, size, price
                self.placeOrder(self.nextValidOrderId, self.contract, _order)
            elif _type == "BKT":
                _price = float(q[5])
                _profit = float(q[6])
                _loss = float(q[7])
                _isProfit = q[8]
                _isTrail = q[9]
                _isLossLimit = q[10]
                _orders = self.BracketOrder(self.nextOrderId(), _action, _size,
                                            _price, _profit, _loss,
                                            _isProfit, _isTrail, _isLossLimit)
                for _ord in _orders:
                    self.placeOrder(_ord.orderId, self.contract, _ord)
                    self.nextOrderId()

            if _type != "BKT":
                self.nextValidOrderId += 1

    def req_CancelOrder(self, q: Queue):
        """Order Cancel requested"""
        #if not self.isOrderCompleted:  # todo order completed
        if q[1] == "GLOBAL":
            self.reqGlobalCancel()
        elif q[1] == "BKT":
            logging.info("Cancel (bracket) Order requested")
            if self.activeOrderId is not None:
                self.cancelOrder(self.activeOrderId)
        else:
            logging.info("Cancel Order requested")
            if self.activeOrderId is not None:
                self.cancelOrder(self.activeOrderId)

    def req_Exit(self, q: Queue):
        """Exit process requested"""
        self.exitReq = True
        self.inQ.put_nowait(q)
        self.outMLQ.put_nowait(("EXIT",))
        self.req_AccountPortfolioStop()
        self.req_Level1Stop()
        self.req_Level2Stop()
        if self.isScanning:
            logging.info("Scanner stopping...")
            self.cancelScannerSubscription(self.scannerId)
            for ind, ticker in enumerate(self.scanList):  # cancel market data for every scan ticker
                if ticker != "*":
                    self.cancelMktData(self.scannerDataId + ind)

        self.disconnect()
        logging.info("Disconnected->Exiting")

    def start(self):
        """Start Process requested: Read Queue until get "connect"=run main "run" function/loop or "Exit" """
        while not self.connectReq:
            self.readQ()
            if self.exitReq:
                return
        #print("Starting main IB API run loop...")
        logging.info("Running IBAPI main loop")
        self.run()  # main loop with modified run to read in_Queue
        #print("Stopped main IB API run loop")
        logging.info("Finished running IBAPI")

    def connect(self, host="127.0.0.1", port=DEFAULT_PORT, clientId=0):
        """Overridden EClient: connect to TWS"""
        super().connect(host, port, clientId)
        logging.info("Connecting to TWS")

    def connectAck(self):
        """ Overridden EWrapper = callback signifying completion of successful connection """
        super().connectAck()
        q = ("CONNECTED",)
        self.connectionOK = True
        self.outQ.put_nowait(q)
        # q = ("LOG", str("Connected to TWS"))
        # self.outQ.put_nowait(q)
        logging.info("Connected to TWS")
        #print("Connection to TWS established")

    def nextValidId(self, orderId: int):
        """  Overridden EWrapper: Receives next valid order id."""
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        self.connectionOK = True
        logging.info("Next Valid ID: " + str(orderId))

    def nextOrderId(self):
        oid = self.nextValidOrderId
        self.nextValidOrderId += 1
        return oid

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        """Overridden EWrapper: receives contract details"""
        super().contractDetails(reqId, contractDetails)

        self.newscontractId = contractDetails.contract.conId
        self.longName = contractDetails.longName
        self.validExchanges = contractDetails.validExchanges
        self.priceMinIncrement = contractDetails.minTick

        # print("Contract Details: " +
        #       " Valid Exchanges: " + str(contractDetails.validExchanges) +
        #       " Contract ID: " + str(self.newscontractId) +
        #       " minprice: " + str(contractDetails.minTick) +
        #       " Long Name: " + str(contractDetails.longName))

    def contractDetailsEnd(self, reqId: int):
        """Overridden EWrapper: receives contract details end signal"""
        super().contractDetailsEnd(reqId)

        q = ("COMPANY_INFO_END",
             self.longName+"@"+str(self.priceMinIncrement), self.priceMinIncrement)
        self.outQ.put_nowait(q)

        self.reqHistoricalNews(self.newsId, self.newscontractId, "BRFG+BRFUPDN+DJNL", "", "", NEWS_NUMBER, [])

        logging.info("Company Info Ended/Sent")
        #print("ContractDetailsEnd. ReqId:", reqId)

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Overridden EWrapper: receives account value"""
        # updates every three minutes if no changes
        #super().updateAccountValue(key, val, currency, accountName)

        if self.isAccPortStreaming:
            if currency == "USD":
                if key == "NetLiquidationByCurrency":
                    self.NetLiquidationByCurrency = float(val)  # total account value = cash+stocks
                elif key == "TotalCashBalance":
                    self.TotalCashBalance = float(val)  # free cash available
                elif key == "StockMarketValue":
                    self.StockMarketValue = float(val)  # stocks value
                elif key == "RealizedPnL":
                    self.RealizedPnL = float(val)  # stocks realized PnL (total of a day?)
                elif key == "UnrealizedPnL":
                    self.UnrealizedPnL = float(val)  # stoccks unrealized PnL (total of a day?)
                q = ("ACCOUNT_UPDATE",
                     self.NetLiquidationByCurrency,
                     self.TotalCashBalance,
                     self.StockMarketValue,
                     self.RealizedPnL+self.UnrealizedPnL)
                self.outQ.put_nowait(q)
                logging.debug("Account Updated")

            # print("UpdateAccountValue. Key:", key, "Value:", val,
            #       "Currency:", currency, "AccountName:", accountName)

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float,
                        realizedPNL: float, accountName: str):
        """Overridden EWrapper: receives portfolio updates value"""
        # updates every three minutes if no changes
        super().updatePortfolio(contract, position, marketPrice, marketValue,
                                averageCost, unrealizedPNL, realizedPNL, accountName)
        if self.isAccPortStreaming:
            if contract.secType == "STK":  # omit closed positions
                self.posSize = position  # number of shares
                self.posPrice = marketPrice  # current price per share
                self.posCost = averageCost  # average paid per share
                self.posRealizedPnL = realizedPNL  # realized total PnL
                self.posUnrealizedPnL = unrealizedPNL  # unrealized total PnL
                _data = np.array([self.posSize,
                                  self.posPrice,
                                  self.posCost,
                                  self.posUnrealizedPnL,
                                  contract.symbol])
                q = ("PORTFOLIO_UPDATE", _data)
                self.outQ.put_nowait(q)

                logging.info("Portfolio Updated")
        #
                # print("UpdatePortfolio.", "Symbol:", contract.symbol, "SecType:", contract.secType,
                #       "Position:", position, "MarketPrice:", marketPrice,
                #       "MarketValue:", marketValue, "AverageCost:", averageCost,
                #       "UnrealizedPNL:", unrealizedPNL, "RealizedPNL:", realizedPNL,
                #       "AccountName:", accountName)

    def accountDownloadEnd(self, account: str):
        # q = ("PORTFOLIO_UPDATE", self.positionData)
        # self.outQ.put_nowait(q)
        logging.info("accountDownloadEnd")

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        # updates every three minutes if no changes
        super().position(account, contract, position, avgCost)
        if self.isAccPortStreaming:  # omit closed positions:
            #
            # self.posSize = position  # number of shares
            # self.posCost = avgCost  # average paid per share
            #contract.symbol
            print("Position.", "Account:", account, "Symbol:", contract.symbol, "SecType:", contract.secType,
                  "Currency:", contract.currency, "Position:", position, "Avg cost:", avgCost)
            print("Position updated")
            logging.debug("Position Updated")

    def positionEnd(self):
        super().positionEnd()
        if self.isAccPortStreaming:
            # q = ("POSITION_UPDATE", self.posSize, self.posCost, contract.symbol)
            # self.outQ.put_nowait(q)
            logging.debug("Position Update Ended/Sent")
            print("PositionEnd Received/Sent")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order,
                  orderState: OrderState):
        super().openOrder(orderId, contract, order, orderState)
        # print("OpenOrder. PermId: ", order.permId, "ClientId:", order.clientId, " OrderId:", orderId,
        #       "Account:", order.account, "Symbol:", contract.symbol, "SecType:", contract.secType,
        #       "Exchange:", contract.exchange, "Action:", order.action, "OrderType:", order.orderType,
        #       "TotalQty:", order.totalQuantity, "CashQty:", order.cashQty,
        #       "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status)
        self.activeOrderId = orderId
        q = (" OrderId:", orderId, "Status:", orderState.status,  # 1,3
             "Symbol:", contract.symbol, "Action:", order.action,  # 5,7
             "OrderType:", order.orderType, "TotalQty:", order.totalQuantity,  # 9, 11
             "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice)  # 13, 15
        self.outQ.put_nowait(("ORDER", "OPEN", q))

    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                    remaining: float, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float):
        super().orderStatus(orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        # print("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled,
        #       "Remaining:", remaining, "AvgFillPrice:", avgFillPrice,
        #       "PermId:", permId, "ParentId:", parentId, "LastFillPrice:",
        #       lastFillPrice, "ClientId:", clientId, "WhyHeld:",
        #       whyHeld, "MktCapPrice:", mktCapPrice)
        q = ("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled, "Remaining:", remaining,  # 1,3,5,7
             "AvgFillPrice:", avgFillPrice, "LastFillPrice:", lastFillPrice)  # 9,11
        self.outQ.put_nowait(("ORDER", "STATUS", q))

    def completedOrder(self, contract: Contract, order: Order,
                       orderState: OrderState):
        super().completedOrder(contract, order, orderState)
        # print("CompletedOrder. PermId:", order.permId, "ParentPermId:", utils.longToStr(order.parentPermId), "Account:", order.account,
        #       "Symbol:", contract.symbol, "SecType:", contract.secType, "Exchange:", contract.exchange,
        #       "Action:", order.action, "OrderType:", order.orderType, "TotalQty:", order.totalQuantity,
        #       "CashQty:", order.cashQty, "FilledQty:", order.filledQuantity,
        #       "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status,
        #       "Completed time:", orderState.completedTime, "Completed Status:" + orderState.completedStatus)
        q = ("CompletedOrder. PermId:", order.permId,
             "Symbol:", contract.symbol, "Action:", order.action, "OrderType:", order.orderType,
             "TotalQty:", order.totalQuantity, "FilledQty:", order.filledQuantity,
             "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status,
             "Completed time:", orderState.completedTime, "Completed Status:" + orderState.completedStatus)
        self.outQ.put_nowait(("ORDER", "COMPLETED", q))

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        super().execDetails(reqId, contract, execution)
        # print("ExecDetails. ReqId:", reqId, "Symbol:", contract.symbol, "SecType:", contract.secType, "Currency:", contract.currency, execution)
        q = ("ExecDetails. ReqId:", reqId, "Symbol:", contract.symbol,
             "Side:", execution.side, "Number of Shares", execution.shares, "Price:", execution.price)
        self.outQ.put_nowait(("ORDER", "EXECDETAILS", q))

    def scannerParameters(self, xml: str):
        super().scannerParameters(xml)
        open('log/scanner.xml', 'w').write(xml)
        print("ScannerParameters received.")

    def scannerData(self, reqId: int, rank: int, contractDetails: ContractDetails,
                    distance: str, benchmark: str, projection: str, legsStr: str):
        super().scannerData(reqId, rank, contractDetails, distance, benchmark,
                            projection, legsStr)
        #        print("ScannerData. ReqId:", reqId, "Rank:", rank, "Symbol:", contractDetails.contract.symbol,
        #              "SecType:", contractDetails.contract.secType,
        #              "Currency:", contractDetails.contract.currency,
        #              "Distance:", distance, "Benchmark:", benchmark,
        #              "Projection:", projection, "Legs String:", legsStr)
        self.scanList = np.append(self.scanList, contractDetails.contract.symbol)
        # print("ScannerData. ReqId:", reqId,
        #       ScanData(contractDetails.contract, rank, distance, benchmark, projection, legsStr))
        logging.debug("Scanner List Updated")

    def scannerDataEnd(self, reqId: int):
        super().scannerDataEnd(reqId)

        self.cancelScannerSubscription(self.scannerId)  # stop new list to have time for info collection
        self.scanData = np.zeros((self.scanList.size, 9), float)  # create array to fill with market data
        for ind, ticker in enumerate(self.scanList):  # request data for every scan ticker
            if ticker != "*":
                _contract = Contract()
                _contract.symbol = ticker
                _contract.secType = "STK"
                _contract.currency = "USD"
                _contract.exchange = "SMART"
                #self.reqMktData(self.scannerDataId + ind, _contract, "165", False, False, [])
                self.reqMktData(self.scannerDataId + ind, _contract, "165, 233", False, False, [])

        self.last_Scan_time = tm.time()
        #print("ScannerListEnd. ReqId:", reqId)
        logging.info("ScannerList End")

    def scannerDataSend(self, isFull, tickerId=0):

        if isFull:
            for ind, ticker in enumerate(self.scanList):  # cancel market data for every scan ticker
                if ticker != "*":
                    self.cancelMktData(self.scannerDataId+ind)

            q = ("SCANNER_FULL", np.nan_to_num(self.scanList), np.nan_to_num(self.scanData))
            self.outL1Q.put_nowait(q)
            self.isScanning = False
            self.scanList = np.array([])  # clean for next scan list/update
            self.scanData = np.zeros((0, 9), float)
            self.isScanDataComplete = False

            logging.info("ScannerData Full completed/Sent")
        else:
            q = ("SCANNER", self.scanList, self.scanData[tickerId, :], self.scanList[tickerId])
            self.outL1Q.put_nowait(q)

            logging.debug("ScannerData Partial completed/Sent")

    # def newsProviders(self, newsProviders: ListOfNewsProviders):
    #     pass
    #     #print("NewsProviders: ")
    #     # for provider in newsProviders:
    #     #     print("NewsProvider.", provider)

    def historicalNews(self, reqId: int, time: str, providerCode: str,
                       articleId: str, headline: str):
        # print("HistoricalNews. ReqId:", reqId, "Time:", time,
        #       "ProviderCode:", providerCode, "ArticleId:", articleId,
        #       "Headline:", headline)
        self.newsData.append(str(time+headline))
        logging.debug("News Data Updated")

    def historicalNewsEnd(self, reqId:int, hasMore:bool):
        #print("HistoricalNewsEnd. ReqId:", reqId, "HasMore:", hasMore)

        q = ("NEWS", self.newsData)
        self.outQ.put_nowait(q)
        self.newsData = []  # clean for next scan list/update

        logging.info("News Data Completed/Sent")

    def mktDepthExchanges(self, depthMktDataDescriptions:ListOfDepthExchanges):
        super().mktDepthExchanges(depthMktDataDescriptions)
        print("MktDepthExchanges:")
        for desc in depthMktDataDescriptions:
            print("DepthMktDataDescription.", desc)

    def historicalData(self, reqId: int, bar: BarData):
        """Overridden EWrapper: receives historical data"""
        #super().historicalData(reqId, bar)
        _date = int(bar.date)
        if reqId == self.slowHistoryId:  # slow trades
            if self.data_slowCandles.size == 0:
                self.data_slowCandles = np.array([int(bar.date), bar.open, bar.close, bar.low, bar.high])
                self.data_slowBars = np.array([int(bar.date), bar.volume])
                self.data_slowCurve = np.array([int(bar.date), bar.barCount])
            else:
                self.data_slowCandles = np.vstack((self.data_slowCandles, [int(bar.date), bar.open, bar.close, bar.low, bar.high]))
                self.data_slowBars = np.vstack((self.data_slowBars, [int(bar.date), bar.volume]))
                self.data_slowCurve = np.vstack((self.data_slowCurve, [int(bar.date), bar.barCount]))
            logging.debug("Slow Historical data received")
            #print("Slow HistoricalData. ReqId:", reqId, "BarData.", bar)
        if reqId == self.HighLowHistoryId:
            if self.data_1min_HighLows.size == 0:
                self.data_1min = np.array([[int(bar.date), bar.open, bar.high, bar.low, bar.close,
                                           bar.volume, bar.barCount, bar.average]])
                self.data_1min_HighLows = np.array([[int(bar.date), bar.high, bar.low]])
            else:
                self.data_1min = np.vstack((self.data_1min, [int(bar.date), bar.open, bar.high, bar.low, bar.close,
                                                             bar.volume, bar.barCount, bar.average]))
                self.data_1min_HighLows = np.vstack((self.data_1min_HighLows, [int(bar.date), bar.high, bar.low]))
            logging.debug("HighLow Historical data received")
        elif reqId == self.fastHistoryId:  # fast trades
            if self.data_fastCandles.size == 0:
                self.data_fastCandles = np.array([int(bar.date), bar.open, bar.close, bar.low, bar.high])
                self.data_fastBars = np.array([int(bar.date), bar.volume])
                self.data_fastCurve = np.array([int(bar.date), bar.barCount])
            else:
                self.data_fastCandles = np.vstack((self.data_fastCandles, [int(bar.date), bar.open, bar.close, bar.low, bar.high]))
                self.data_fastBars = np.vstack((self.data_fastBars, [int(bar.date), bar.volume]))
                self.data_fastCurve = np.vstack((self.data_fastCurve, [int(bar.date), bar.barCount]))
            logging.debug("Fast Historical data received")
            #print("Fast HistoricalData Trades. ReqId:", reqId, "BarData.", bar)
        elif reqId == self.fastHistoryId_bid:  # bid update
            if self.data_fastCurveBid.size == 0:
                self.data_fastCurveBid = np.array([int(bar.date), bar.low])
            else:
                self.data_fastCurveBid = np.vstack((self.data_fastCurveBid, [int(bar.date), bar.low]))
            logging.debug("Fast Historical Bid received")
            #print("Fast HistoricalData Bid. ReqId:", reqId, "BarData.", bar)
        elif reqId == self.fastHistoryId_ask:  # ask update
            if self.data_fastCurveAsk.size == 0:
                self.data_fastCurveAsk = np.array([int(bar.date), bar.high])
            else:
                self.data_fastCurveAsk = np.vstack((self.data_fastCurveAsk, [int(bar.date), bar.high]))
            logging.debug("Fast Historical Ask received")
            #print("Fast HistoricalData Ask. ReqId:", reqId, "BarData.", bar)

    def historicalDataUpdate(self, reqId: int, bar: BarData):
        """Overridden EWrapper: receives historical data update"""
        #super().historicalDataUpdate(reqId, bar)
        #print("Slow HistoricalData update. ReqId:", reqId, bar)
        if self.isLevel1Streaming:
            if reqId == self.slowHistoryId:# and bar.open > 0 and bar.close > 0 and bar.low > 0 and bar.high > 0 and bar.volume >= 0:
                self.data_slowCandles = np.array([int(bar.date), bar.open, bar.close, bar.low, bar.high])
                self.data_slowBars = np.array([int(bar.date), bar.volume])
                self.data_slowCurve = np.array([int(bar.date), bar.barCount])
                logging.debug("Slow Historical data update received")
                #print("Slow HistoricalData Update. ReqId:", reqId, "BarData.", bar)
                q = ("SLOW_HISTORY_UPDATE", self.data_slowCandles, self.data_slowBars, self.data_slowCurve)
                self.outL1Q.put_nowait(q)
            if reqId == self.HighLowHistoryId:

                if self.data_1min_HighLows.size == 0:  # in case somehow update before data arrived
                    self.data_1min = np.array([[int(bar.date), bar.open, bar.high, bar.low, bar.close,
                                               bar.volume, bar.barCount, bar.average]])
                    self.data_1min_HighLows = np.array([[int(bar.date), bar.high, bar.low]])

                else:
                    if self.data_1min_HighLows[-1, 0] == int(bar.date):  # update
                        self.data_1min[-1, 1:] = np.array([[bar.open, bar.high, bar.low, bar.close,
                                                                     bar.volume, bar.barCount, bar.average]])
                        self.data_1min_HighLows[-1, 1:] = np.array([[bar.high, bar.low]])
                    else:  # or add
                        self.data_1min = np.vstack((self.data_1min, [int(bar.date), bar.open, bar.high, bar.low, bar.close,
                                                                     bar.volume, bar.barCount, bar.average]))
                        self.data_1min_HighLows = np.vstack((self.data_1min_HighLows, [int(bar.date), bar.high, bar.low]))

                if self.data_1min.size > 8:  # more than one record
                    isUp = self.data_1min[-1, 4] > self.data_1min[-2, 4]
                else:
                    isUp = False

                _highlow = np.zeros((2, 4))
                if self.data_1min_HighLows.size >= 3:  # i min current
                    _highlow[:, 3] = self.data_1min_HighLows[-1, 1:]
                if self.data_1min_HighLows.size >= 6:  # i min previous
                    _highlow[:, 2] = self.data_1min_HighLows[-2, 1:]
                if self.data_1min_HighLows.size >= 15:  # 5 min current
                    _highlow[0, 1] = max(self.data_1min_HighLows[-5:, 1])  # high
                    _highlow[1, 1] = min(self.data_1min_HighLows[-5:, 2])  # low
                if self.data_1min_HighLows.size >= 30:  # 5 min current
                    _highlow[0, 0] = max(self.data_1min_HighLows[-10:-5, 1])  # high
                    _highlow[1, 0] = min(self.data_1min_HighLows[-10:-5, 2])  # low

                q = ("HIGHLOW_UPDATE", _highlow, isUp)
                self.outL1Q.put_nowait(q)
                q2 = ("DATA_UPDATE", self.data_1min)
                self.outMLQ.put_nowait(q2)

                logging.debug("HighLow Historical data updated")
            elif reqId == self.fastHistoryId:# and bar.open > 0 and bar.close > 0 and bar.low > 0 and bar.high > 0 and bar.volume >= 0:
                self.data_fastCandles = np.array([int(bar.date), bar.open, bar.close, bar.low, bar.high])
                self.data_fastBars = np.array([int(bar.date), bar.volume])
                self.data_fastCurve = np.array([int(bar.date), bar.barCount])
                logging.debug("Fast Historical data Trades update received")
                #print("Fast HistoricalData Update. ReqId:", reqId, "BarData.", bar)
                q = ("FAST_HISTORY_UPDATE", "TRADES", self.data_fastCandles, self.data_fastBars, self.data_fastCurve)
                self.outL1Q.put_nowait(q)
            elif reqId == self.fastHistoryId_bid:# and bar.low > 0: # history end sent
                self.data_fastCurveBid = np.array([int(bar.date), bar.low])
                logging.debug("Fast Historical data Bid update received")
                #print("Fast HistoricalData Update. ReqId:", reqId, "BarData.", bar)
                q = ("FAST_HISTORY_UPDATE", "BID", self.data_fastCurveBid)
                self.outL1Q.put_nowait(q)
            elif reqId == self.fastHistoryId_ask:# and bar.high > 0: # history end sent
                self.data_fastCurveAsk = np.array([int(bar.date), bar.high])
                logging.debug("Fast Historical data Ask update received")
                # print("Fast HistoricalData Update. ReqId:", reqId, "BarData.", bar)
                q = ("FAST_HISTORY_UPDATE", "ASK", self.data_fastCurveAsk)
                self.outL1Q.put_nowait(q)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Overridden EWrapper: receives historical data end signal"""
        super().historicalDataEnd(reqId, start, end)

        if reqId == self.slowHistoryId:
            q = ("SLOW_HISTORY_END", self.data_slowCandles, self.data_slowBars, self.data_slowCurve)
            self.outL1Q.put_nowait(q)
            logging.info("Slow Historical data Ended/Sent")
            #print("Slow HistoricalData End. ReqId:", reqId, "from", start, "to", end)
        if reqId == self.HighLowHistoryId:
            _highlow = np.zeros((2, 4))
            if self.data_1min_HighLows.size >= 3:  # i min current
                _highlow[:, 3] = self.data_1min_HighLows[-1, 1:]
            if self.data_1min_HighLows.size >= 6:  # i min previous
                _highlow[:, 2] = self.data_1min_HighLows[-2, 1:]
            if self.data_1min_HighLows.size >= 15:  # 5 min current
                _highlow[0, 1] = max(self.data_1min_HighLows[-5:, 1])  # high
                _highlow[1, 1] = min(self.data_1min_HighLows[-5:, 2])  # low
            if self.data_1min_HighLows.size >= 30:  # 5 min current
                _highlow[0, 0] = max(self.data_1min_HighLows[-10:-5, 1])  # high
                _highlow[1, 0] = min(self.data_1min_HighLows[-10:-5, 2])  # low

            q = ("HIGHLOW_UPDATE", _highlow, True)
            self.outL1Q.put_nowait(q)
            q2 = ("DATA_END", self.data_1min)
            self.outMLQ.put_nowait(q2)

            logging.info("HighLow Historical data Ended/Sent")
        elif reqId == self.fastHistoryId:
            q = ("FAST_HISTORY_END", "TRADES", self.data_fastCandles, self.data_fastBars, self.data_fastCurve)
            self.outL1Q.put_nowait(q)
            logging.info("Fast Historical data Trades Ended/Sent")
            #print("Fast HistoricalData Trades End. ReqId:", reqId, "from", start, "to", end)
        elif reqId == self.fastHistoryId_bid:
            q = ("FAST_HISTORY_END", "BID", self.data_fastCurveBid)
            self.outL1Q.put_nowait(q)
            logging.info("Fast Historical data Bid Ended/Sent")
            #print("Fast HistoricalData Bid/Ask End. ReqId:", reqId, "from", start, "to", end)

        elif reqId == self.fastHistoryId_ask:
            q = ("FAST_HISTORY_END", "ASK", self.data_fastCurveAsk)
            self.outL1Q.put_nowait(q)
            logging.info("Fast Historical data Ask Ended/Sent")
            #print("Fast HistoricalData Bid/Ask End. ReqId:", reqId, "from", start, "to", end)

    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib):
        """Overridden EWrapper: receives tickPrice update"""
        #super().tickPrice(reqId, tickType, price, attrib)

        if self.isLevel1Streaming and reqId == self.streamLevel1Id:
            if tickType in [1, 2, 4, 5, 6, 7, 8, 9, 56]:# and price > 0: # history end sent
                self.data_tick = np.array([tickType, price])
                q = ("LEVEL1_UPDATE", self.data_tick)
                self.outL1Q.put_nowait(q)
                logging.debug("Tick Price update received")
                # print("TickPrice. TickerId:", reqId, "tickType:", tickType,
                #       "Price:", price, "CanAutoExecute:", attrib.canAutoExecute,
                #       "PastLimit:", attrib.pastLimit, end=' ')

        if (self.scannerDataId + self.scanList.size) > reqId >= self.scannerDataId:  # data from scanner
            _ind = reqId - self.scannerDataId
            if tickType == 4:  # last price
                self.scanData[_ind, 0] = abs(price)
            elif tickType == 9:  # previous day close
                self.scanData[_ind, 2] = abs(price)
            elif tickType == 1:  # bid price
                self.scanData[_ind, 3] = abs(price)
            elif tickType == 2:  # ask price
                self.scanData[_ind, 4] = abs(price)
            elif tickType == 6:  # HoD
                self.scanData[_ind, 6] = abs(price)
            elif tickType == 7:  # LoD
                self.scanData[_ind, 7] = abs(price)

            if tickType in [1, 2, 4, 6, 7, 9]:
                self.isScanDataComplete = np.all(self.scanData[self.scanList != "*", :] >= 0.1)

                logging.debug("Scanner List Updated")
                if self.isScanDataComplete or tm.time() - self.last_Scan_time > MAX_SCAN_DURATION:
                    self.scannerDataSend(True)
                elif np.all(self.scanData[_ind] >= 0.1):
                    self.scannerDataSend(False, _ind)

    def tickSize(self, reqId: TickerId, tickType: TickType, size: int):
        """Overridden EWrapper: receives tickSize update"""
        #super().tickSize(reqId, tickType, size)
        if self.isLevel1Streaming and reqId == self.streamLevel1Id:
            if tickType in [1, 2, 4, 5, 6, 7, 8, 9, 21, 56]:  # and size >= 0: # history end sent
                self.data_tick = np.array([tickType, size])
                q = ("LEVEL1_UPDATE", self.data_tick)
                self.outL1Q.put_nowait(q)
                logging.debug("Tick Size update received")
                #print("TickSize. TickerId:", reqId, "TickType:", tickType, "Size:", size)

        if (self.scannerDataId + self.scanList.size) > reqId >= self.scannerDataId:  # data from scanner
            _ind = reqId - self.scannerDataId
            if tickType == 8:  # volume of the day
                self.scanData[_ind, 1] = abs(size)
            elif tickType == 21:  # average volume over 90 days
                self.scanData[_ind, 5] = abs(size)

            if tickType in [8, 21]:
                self.isScanDataComplete = np.all(self.scanData[self.scanList != "*", :] >= 0.1)

                logging.debug("Scanner List Updated")
                if self.isScanDataComplete or tm.time() - self.last_Scan_time > MAX_SCAN_DURATION:
                    self.scannerDataSend(True)
                elif np.all(self.scanData[_ind] >= 0.1):
                    self.scannerDataSend(False, _ind)

    def tickGeneric(self, reqId: TickerId, tickType: TickType, value: float):
        """Overridden EWrapper: receives tickGeneric update"""
        #super().tickGeneric(reqId, tickType, value)

        if self.isLevel1Streaming:
            if tickType in [1, 2, 4, 5, 6, 7, 8, 9, 56]:
                self.data_tick = np.array([tickType, value])
                q = ("LEVEL1_UPDATE", self.data_tick)
                self.outL1Q.put_nowait(q)
                logging.debug("Tick Generic update received")
                #print("TickGeneric. TickerId:", reqId, "TickType:", tickType, "Value:", value)

    def tickString(self, reqId: TickerId, tickType: TickType, value: str):
        # super().tickString(reqId, tickType, value)
        # print("TickString. TickerId:", reqId, "Type:", tickType, "Value:", value)

        if (self.scannerDataId + self.scanList.size) > reqId >= self.scannerDataId:  # data from scanner
            _ind = reqId - self.scannerDataId
            if tickType == 48:  # RT Volume
                try:
                    self.scanData[_ind, 8] = float(value.split(";")[4])
                except:
                    pass

                # last trade's price, size and time along with current day's total traded volume,
                # Volume Weighted Average Price (VWAP) and whether or not the trade was filled by a single market maker
                # Example: 701.28;1;1348075471534;67854;701.46918464;true
                self.isScanDataComplete = np.all(self.scanData[self.scanList != "*", :] >= 0.1)

                logging.debug("Scanner List Updated")
                if self.isScanDataComplete or tm.time() - self.last_Scan_time > MAX_SCAN_DURATION:
                    self.scannerDataSend(True)
                elif np.all(self.scanData[_ind] >= 0.1):
                    self.scannerDataSend(False, _ind)

    def tickByTickAllLast(self, reqId: int, tickType: int, time, price: float,
                          size: int, tickAtrribLast: TickAttribLast, exchange: str,
                          specialConditions: str):
        """Overridden EWrapper: receives AllLastTickTick update"""
        # super().tickByTickAllLast(reqId, tickType, time, price, size, tickAtrribLast,
        #                           exchange, specialConditions)
        if self.isLevel2Streaming:
            #_start = tm.perf_counter()
            self.data_level2Scatter_ticks.append(np.array([tm.time(), price, size]))

            logging.debug("Tick-by-Tick Trades update received")
            _end_time = tm.perf_counter()
            #print(f"last time: {_end_time - _start}")
            if abs(_end_time - self.timeLevel2TradesUpdate) > 1 / LEVEL2_QUEUE_RATE:
                q = ("LEVEL2_UPDATE", "TRADES", np.array(self.data_level2Scatter_ticks))
                self.outL2Q.put_nowait(q)
                self.data_level2Scatter_ticks = []  # clear the data [time, price, size]
                self.timeLevel2TradesUpdate = _end_time  # update time
                logging.debug("Trades block update recalculated/sent")
            # print(" ReqId:", reqId,
            #       "Time:", datetime.datetime.fromtimestamp(time).strftime("%Y%m%d %H:%M:%S"),
            #       "Price:", price, "Size:", size, "Exch:", exchange,
            #       "Spec Cond:", specialConditions, "PastLimit:", tickAtrribLast.pastLimit, "Unreported:",
            #       tickAtrribLast.unreported)

    def historicalTicksLast(self, reqId: int, ticks: ListOfHistoricalTickLast, done: bool):
        """Overridden EWrapper: receives AllLastTickTick update"""
        #super().historicalTicksLast(reqId, ticks, done)
        if self.isLevel2Streaming:
            #_start = tm.perf_counter()
            for tick in ticks:
                self.data_level2Scatter_ticks.append(np.array([tick.time, tick.price, tick.size]))

            logging.debug("Tick-by-Tick Trades history received")

            _end_time = tm.perf_counter()
            #print(f"last time: {_end_time - _start}")
            if abs(_end_time - self.timeLevel2TradesUpdate) > 1 / LEVEL2_QUEUE_RATE:
                q = ("LEVEL2_UPDATE", "TRADES", np.array(self.data_level2Scatter_ticks))
                self.outL2Q.put_nowait(q)
                self.data_level2Scatter_ticks = []  # clear the data [time, price, size]
                self.timeLevel2TradesUpdate = _end_time  # update time
                logging.debug("Trades block update recalculated/sent")

    def tickByTickBidAsk(self, reqId: int, time, bidPrice: float, askPrice: float,
                         bidSize: int, askSize: int, tickAttribBidAsk: TickAttribBidAsk):
        """Overridden EWrapper: receives BidAskTickTick update"""
        # super().tickByTickBidAsk(reqId, time, bidPrice, askPrice, bidSize,
        #                          askSize, tickAttribBidAsk)
        if self.isLevel2Streaming:
            #_start = tm.perf_counter()
            self.data_level2BACurves_ticks.append(np.array([tm.time(), bidPrice, askPrice, bidSize, askSize]))

            logging.debug("Tick-by-Tick BidAsk update received")
            _end_time = tm.perf_counter()
            #print(f"BA time: {_end_time - _start}")
            if abs(_end_time - self.timeLevel2BidAskUpdate) > 1 / LEVEL2_QUEUE_RATE:
                q = ("LEVEL2_UPDATE", "BA", np.array(self.data_level2BACurves_ticks))
                self.outL2Q.put_nowait(q)
                self.data_level2BACurves_ticks = []  # clear the data
                self.timeLevel2BidAskUpdate = _end_time  # update time
                logging.debug("BidAsk block update recalculated/sent")

            # print("BidAsk. ReqId:", reqId,
            #       "Time:", datetime.datetime.fromtimestamp(time).strftime("%Y%m%d %H:%M:%S"),
            #       "BidPrice:", bidPrice, "AskPrice:", askPrice, "BidSize:", bidSize,
            #       "AskSize:", askSize, "BidPastLow:", tickAttribBidAsk.bidPastLow, "AskPastHigh:",
            #       tickAttribBidAsk.askPastHigh)

    def historicalTicksBidAsk(self, reqId: int, ticks: ListOfHistoricalTickBidAsk, done: bool):
            """Overridden EWrapper: receives BidAskTickTick history"""
            #super().historicalTicksBidAsk(reqId, ticks, done)
            if self.isLevel2Streaming:
                _start = tm.perf_counter()
                for tick in ticks:
                    self.data_level2BACurves_ticks.append(np.array([tick.time, tick.priceBid, tick.priceAsk, tick.sizeBid,
                                                                    tick.sizeAsk]))

                logging.debug("Tick-by-Tick BidAsk history received")
                _end_time = tm.perf_counter()
                #print(f"BA time: {_end_time-_start}")
                if abs(_end_time - self.timeLevel2BidAskUpdate) > 1 / LEVEL2_QUEUE_RATE:
                    q = ("LEVEL2_UPDATE", "BA", np.array(self.data_level2BACurves_ticks))
                    self.outL2Q.put_nowait(q)
                    self.data_level2BACurves_ticks = []  # clear the data
                    self.timeLevel2BidAskUpdate = _end_time  # update time
                    logging.debug("BidAsk block update recalculated/sent")

    def updateMktDepth(self, reqId: TickerId, position: int, operation: int,
                       side: int, price: float, size: int):
        """Overridden EWrapper: receives MarketDepth update"""
        # super().updateMktDepth(reqId, position, operation, side, price, size)
        if self.isLevel2Streaming:
            _start = tm.perf_counter()
            self.data_level2OB_cache[self.level2OB_cache_index, :] = [side, operation, position, price, size]
            self.level2OB_cache_index += 1

            _end_time = tm.perf_counter()
            if abs(_end_time - self.timeOrderBookUpdate) > 1 / LEVEL2_QUEUE_RATE:
                _dec = self.priceDecimals  # number of decimal places in min price increment

                for record in self.data_level2OB_cache:  # [-2000:, :]:
                    _side = record[0]
                    _operation = record[1]
                    _position = record[2]
                    _price = record[3]
                    _size = record[4]

                    if _side == 1:  # bid
                        if _operation == 2:  # delete
                            self.data_orderBookBidBars[int(_position)] = np.array([0, 0])
                        else:  # insert or update
                            self.data_orderBookBidBars[int(_position)] = np.array([round(_price, _dec), _size])

                    elif _side == 0:  # ask
                        if _operation == 2:  # delete
                            self.data_orderBookAskBars[int(_position)] = np.array([0, 0])
                        else:  # insert or update
                            self.data_orderBookAskBars[int(_position)] = np.array([round(_price, _dec), _size])

                q = ("LEVEL2_UPDATE", "ORDERBOOK_BID", self.data_orderBookBidBars)
                self.outL2Q.put_nowait(q)
                q = ("LEVEL2_UPDATE", "ORDERBOOK_ASK", self.data_orderBookAskBars)
                self.outL2Q.put_nowait(q)

                self.data_level2OB_cache[:self.level2OB_cache_index, :] = 0.
                self.level2OB_cache_index = 0

                self.timeOrderBookUpdate = _end_time
                # print(f"delay: {tm.perf_counter()-_start}")
                logging.debug("Market Depth block update received/sent")
            # print("UpdateMarketDepth. ReqId:", reqId, "Position:", position, "Operation:",
            #       operation, "Side:", side, "Price:", price, "Size:", size)

    def updateMktDepthL2(self, reqId: TickerId, position: int, marketMaker: str,
                         operation: int, side: int, price: float, size: int, isSmartDepth: bool):
        """Overridden EWrapper: receives MarketDepth version 2 update"""
        # super().updateMktDepthL2(reqId, position, marketMaker, operation, side,
        #                          price, size, isSmartDepth)

        if self.isLevel2Streaming and (self.streamNYSE or (not self.streamNYSE and "NYSE" not in marketMaker)):
            #_start = tm.perf_counter()
            self.data_level2OB_cache[self.level2OB_cache_index, :] = [side, operation, position, price, size]
            self.level2OB_cache_index += 1
            _end_time = tm.perf_counter()

            #print(f"ob time: {(_end_time-_start)*1e6}")
            if abs(_end_time - self.timeOrderBookUpdate) > 1 / LEVEL2_QUEUE_RATE:
                _dec = self.priceDecimals  # number of decimal places in min price increment

                for record in self.data_level2OB_cache[:self.level2OB_cache_index, :]:
                    _side = record[0]
                    _operation = record[1]
                    _position = record[2]
                    _price = record[3]
                    _size = record[4]

                    if _side == 1:  # bid
                        if _operation == 2:  # delete
                            self.data_orderBookBidBars[int(_position)] = np.array([0, 0])
                        else:  # insert or update
                            self.data_orderBookBidBars[int(_position)] = np.array([round(_price, _dec), _size])

                    elif _side == 0:  # ask
                        if _operation == 2:  # delete
                            self.data_orderBookAskBars[int(_position)] = np.array([0, 0])
                        else:  # insert or update
                            self.data_orderBookAskBars[int(_position)] = np.array([round(_price, _dec), _size])

                q = ("LEVEL2_UPDATE", "ORDERBOOK_BID", self.data_orderBookBidBars)
                self.outL2Q.put_nowait(q)
                q = ("LEVEL2_UPDATE", "ORDERBOOK_ASK", self.data_orderBookAskBars)
                self.outL2Q.put_nowait(q)

                #print(self.level2OB_cache_index)
                self.data_level2OB_cache[:self.level2OB_cache_index, :] = 0.
                self.level2OB_cache_index = 0

                self.timeOrderBookUpdate = _end_time
                #print(f"delay: {tm.perf_counter()-_start}")
                logging.debug("Market Depth2 block update received/sent")
            # print("UpdateMarketDepthL2. ReqId:", reqId, "Position:", position, "MarketMaker:", marketMaker, "Operation:",
            #       operation, "Side:", side, "Price:", price, "Size:", size, "isSmartDepth:", isSmartDepth)


    def run(self):
        """Overridden EClient: modified to include readQ method within message loop."""
        try:
            while (self.isConnected() or not self.msg_queue.empty()) and not self.exitReq:

                _end_time = tm.perf_counter()
                if abs(_end_time - self.ibAPI_time) > 0.2:  # send every 200 msexsecond
                    q = ("IB_QUEUE_SIZE", self.msg_queue.qsize())
                    #print("queue size", self.msg_queue.qsize())
                    self.outQ.put_nowait(q)
                    self.ibAPI_time = _end_time
                    self.readQ()  # process the incoming messages via the queue

                try:
                    try:
                        text = self.msg_queue.get(block=True, timeout=IB_SERVER_RATE)

                        if len(text) > MAX_MSG_LEN:
                            self.wrapper.error(NO_VALID_ID, BAD_LENGTH.code(),
                                               "%s:%d:%s" % (BAD_LENGTH.msg(), len(text), text))
                            print("IBAPI bad length error")
                            self.disconnect()
                            break
                    except queue.Empty:
                        logger.debug("queue.get: empty")
                    else:
                        fields = comm.read_fields(text)
                        logger.debug("fields %s", fields)
                        self.decoder.interpret(fields)
                except (KeyboardInterrupt, SystemExit):
                    logger.info("detected KeyboardInterrupt, SystemExit")
                    self.keyboardInterrupt()
                    self.keyboardInterruptHard()
                except BadMessage:
                    logger.info("BadMessage")
                    self.conn.disconnect()
                logger.debug("conn:%d queue.sz:%d",
                             self.isConnected(),
                             self.msg_queue.qsize())
        finally:
            self.disconnect()


    @staticmethod
    def MarketOrder(action: str, quantity: float):
        """Market Order Template"""
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        return order

    @staticmethod
    def MidpriceOrder(action: str, quantity: float):  # , priceCap: float):
        """MidPrice Order Template"""
        order = Order()
        order.action = action
        order.orderType = "MIDPRICE"
        order.totalQuantity = quantity
        # order.lmtPrice = priceCap  # optional
        return order

    @staticmethod
    def LimitOrder(action: str, quantity: float, limitPrice: float, outsideRth=OUTSIDE_RTH):
        """Limit Order Template"""
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limitPrice
        order.outsideRth = outsideRth
        return order

    @staticmethod
    def StopOrder(action: str, quantity: float, stopPrice: float):
        order = Order()
        order.action = action
        order.orderType = "STP"
        order.auxPrice = stopPrice
        order.totalQuantity = quantity
        return order

    @staticmethod
    def StopLimit(action: str, quantity: float, stopPrice: float, lmtPriceOffset: float):
        order = Order()
        order.action = action
        order.orderType = "STP LMT"
        order.totalQuantity = quantity
        order.auxPrice = stopPrice
        order.lmtPrice = stopPrice+lmtPriceOffset*(-1 if action == "SELL" else 1)
        return order

    @staticmethod
    def TrailingStop(action: str, quantity: float,
                     basisPrice: float, trailingPercent: float):
        order = Order()
        order.action = action
        order.orderType = "TRAIL"
        order.totalQuantity = quantity
        # stop price if trail is not yet activated
        order.trailStopPrice = round(basisPrice*(1+trailingPercent*(0.01 if action == "BUY" else -0.01)), 2)
        order.trailingPercent = trailingPercent
        return order

    @staticmethod
    def TrailingStopLimit(action: str, quantity: float,
                          basisPrice: float, trailingPercent: float, lmtPriceOffset: float):
        order = Order()
        order.action = action
        order.orderType = "TRAIL LIMIT"
        order.totalQuantity = quantity
        # stop price if trail is not yet activated
        order.trailStopPrice = round(basisPrice*(1+trailingPercent*(0.01 if action == "BUY" else -0.01)), 2)
        order.trailingPercent = trailingPercent
        order.lmtPriceOffset = lmtPriceOffset
        return order

    #@staticmethod
    def BracketOrder(self, parentOrderId: int, action: str, quantity: float,
                     limitPrice: float, takeProfitLimitPrice: float, stopLossPrice: float,
                     isProfit: bool, isTrail: bool, isLossLimit: bool, useRTH = OUTSIDE_RTH):

        # This will be our main or "parent" order
        # The parent and children orders will need this attribute set to False to prevent accidental executions.
        # The LAST CHILD will have it set to True,
        _nchild = 0
        _limitOffset = 0.0  # positive => worst price, negative => better price/// 0.0 seems to be ok even for fast moving price
        _bracket_action = "SELL" if action == "BUY" else "BUY"

        parent = self.LimitOrder(action, quantity, limitPrice, useRTH)
        parent.orderId = parentOrderId
        parent.transmit = False

        if isProfit:
            takeProfit = self.LimitOrder(_bracket_action, quantity,
                                         takeProfitLimitPrice, useRTH)
            takeProfit.parentId = parentOrderId
            _nchild += 1
            takeProfit.orderId = parent.orderId + _nchild
            takeProfit.ocaType = 2  # to change size depending on other child
            takeProfit.transmit = False


        if isTrail:  # loss-price in percents
            if isLossLimit:
                stopLoss = self.TrailingStopLimit(_bracket_action, quantity,
                                                  limitPrice, stopLossPrice, _limitOffset)
            else:
                stopLoss = self.TrailingStop(_bracket_action, quantity,
                                             limitPrice, stopLossPrice)
        else:  # loss-price in absolute units
            if isLossLimit:
                stopLoss = self.StopLimit(_bracket_action, quantity,
                                          stopLossPrice, _limitOffset)
            else:
                stopLoss = self.StopOrder(_bracket_action, quantity,
                                          stopLossPrice)
        stopLoss.parentId = parentOrderId
        _nchild += 1
        stopLoss.orderId = parent.orderId + _nchild
        stopLoss.ocaType = 2  # to change size depending on other child
        stopLoss.transmit = True

        if _nchild == 1:
            bracketOrder = [parent, stopLoss]
        else:
            bracketOrder = [parent, takeProfit, stopLoss]
        return bracketOrder


def main():
    # app = IbApp()
    # app.connect("127.0.0.1", 7497, 0)
    # app.run()
    pass


if __name__ == "__main__":
    main()

