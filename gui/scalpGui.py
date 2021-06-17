# -*- coding: utf-8 -*-
"""
Module for custom gui window
"""


from PyQt5 import QtWidgets, QtGui, QtCore,  uic
from PyQt5.QtCore import QRectF
import pyqtgraph as pg

import numpy as np
from scipy.interpolate import interp1d

import time, datetime, copy
import logging

from multiprocessing import Queue
import queue

import gui.colors as cl
import tech_analysis as ta
from gui.widgets import CandlestickItem
from gui.widgets import PercentAxisItem
from gui import DateAxisItem_custom as DateAxisItem
from gui.color_constants import *

FRONT_UI_FILE = "gui/scalpGui.ui"

CAPITAL_FRACTION = 0.9  # maximum fraction of capital to be used as stocks
AUTOTRADE_SELL_FRACTION = 1

TRADES_SIZE_NORM = 0.002  # max= percent of price
TRADES_BARS_OPACITY = 0.7

# PERFORMANCE CONSTANTS
LEVEL2_IMAGE_TYPE = float
LEVEL2_IMAGE_SHAPE_TICKS = (3001, 3000)   # (divisable by 4 +1, divisable by 4)
LEVEL2_IMAGE_SHAPE = LEVEL2_IMAGE_SHAPE_TICKS
LEVEL2_IMAGE_SHAPE_VIEW = (1001, 1000)

TRADES_GUI_BARS_NUMBER = 100  # maximum number of Trades(Bars) to plot
TRADES_GUI_NUMBER = 3000  # maximum number of Trades(curve) to plot  (=5 min for 100ms)
TRADES_CALC_NUMBER = 3000  # maximum number of Trades level2 data to recalculate in volume profile  (~25-35ms for full)
BA_CALC_NUMBER = 10000  # maximum number of BidAsks/imbalances level2 data to recalculate
CUMUL_CALC_NUMBER = 10000  # maximum number of BidAsks/imbalances level2 data to recalculate
BA_GUI_NUMBER = 3000  # maximum number of BidAsks/imbalances level2 data to plot (=5 min for 100ms)
FAST_GUI_NUMBER = 1000  # maximum number of fast history candles/volume/etc data to plot
VOL_PROF_CALC_TIME = 3  # time for volume profile recalcualtion in seconds

FRAME_RATE_GUI = 5  # defines minimum time for figures update (not to waste time on too frequent updates)
CROSSHAIR_RATE_GUI = 20
CANDLES_UPDATE_LIMIT = 20  # min number of candles for frequent updates, 2x= maximum

MARKET_DEPTH_LEVELS = 200  # maximum number of market depth levels per bid/ask
GUI_QUEUE_RATE = 0.001  # reading time of queues in seconds

logger = logging.getLogger(__name__)

pg.setConfigOptions(antialias=True)
pg.setConfigOptions(imageAxisOrder="row-major")
pg.setConfigOption("foreground", LGREY)
pg.setConfigOption("background", DGREY)
#pg.setConfigOptions(useOpenGL=True)
#pg.setConfigOptions(enableExperimental=True)

class GuiWindow(QtWidgets.QMainWindow):
    def __init__(self, inQ: Queue, inL1Q: Queue, inL2Q: Queue, inMLQ: Queue, outQ: Queue):
        super().__init__()
        self.app = QtWidgets.QApplication.instance()
        #self.event_loop = QtCore.QEventLoop()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(GUI_QUEUE_RATE*1000)  # resolution is only 1ms, thus either 0 or >1ms
        # ---- pre-init gui
        logging.debug("Loading GUI.ui file")
        uic.loadUi(FRONT_UI_FILE, baseinstance=self)  # Load the main ui file
        logging.debug("Gui window Initialization")
        self.setWindowTitle("Scalp Ray v0.98")
        # _monitor = QtWidgets.QDesktopWidget().screenGeometry(1)
        # self.move(_monitor.left(), _monitor.top())
        self.showMaximized() #showFullScreen() #

        self.utc_offset = 3600
        # --- connection values
        self.inQ = inQ  # from ibapi
        self.inL1Q = inL1Q  # from ibapi
        self.inL2Q = inL2Q  # from ibapi
        self.inMLQ = inMLQ  # from ML
        self.outQ = outQ  # to ibapi
        self.exitReq = False
        self.isConnected = False
        self.ibQueue_size = 0
        #----------------
        self.autotraded = False

        #--------file names
        self.filename_Errors = time.strftime("log/Errors" + "_%y%m%d_%H%M%S.txt")
        self.filename_OrderTape = time.strftime("log/OrderTape" + "_%y%m%d_%H%M%S.txt")
        self.filename_Level2Trades = time.strftime("log/data_dump/Trades" + "_%y%m%d_%H%M%S.dat")
        self.filename_Level2BA = time.strftime("log/data_dump/BidAsks" + "_%y%m%d_%H%M%S.dat")
        self.filename_Level2OrderBook = time.strftime("log/data_dump/OrderBook" + "_%y%m%d_%H%M%S.dat")
        # -------ORder values
        self.last_OrderOpen = None
        self.last_OrderStatus = None
        self.last_avrgSharePrice = 0.01
        self.data_OrderTape = np.array([])  # id*Buy/Sell*Size*Price
        self.data_OrderTape_Ticker_Side = []
        # ---- trading values received
        self.isAccountPortfolioUpdating = False
        self.isLevel1Updating = False
        self.isLevel2Updating = False
        self.priceMinIncrement = 0.01
        self.priceDecimals = 2
        # ----- trading values level1
        self.last_PositionSize = 0
        self.last_Cash = 0
        self.last_Bid = 1.
        self.last_MidPoint = 1.5
        self.last_Ask = 2.
        self.last_Price = 11.
        self.last_High = 1.
        self.last_Low = 1.
        self.last_Close = 1.  # of the previous day
        self.last_Size = 0.  # volume of last trade
        self.last_Volume = 0.
        self.last_avrg_Volume = 0.
        self.last_VolumeRate = 0.
        # ----- trading values level2
        self.last_orderBookBid = []
        self.last_orderBookAsk = []
        # ---- trading values processed
        self.last_sumBidSize = 0
        self.last_sumAskSize = 0
        self.last_sumPriceBidSize = 0
        self.last_sumPriceAskSize = 0
        self.last_avrgBidPrice = 0  # tick updates from Market depth
        self.last_avrgAskPrice = 0  # tick updates from Market depth
        self.last_forceBid = 0
        self.last_forceAsk = 0
        self.last_Spread = 0.  # in percent
        self.last_Change = 0.  # in percent

        #------------- last graph update time and other benchamrks
        now = time.time()
        now_counter = time.perf_counter()
        self.timeUpdate_SlowCandle = now_counter
        self.timeUpdate_SlowVolume = now_counter
        self.timeUpdate_FastCandle = now_counter
        self.timeUpdate_FastVolume = now_counter
        self.timeUpdate_VolumeProf = now_counter
        self.timeUpdate_fullVolumeProf = now_counter
        self.timeUpdate_Level2map = now_counter
        self.timeUpdate_recalcLevel2map = now_counter
        self.timeUpdate_Level2curves = now_counter
        self.timeUpdate_recalcLevel2BA = now_counter
        self.timeUpdate_recalcLevel2Trades = now_counter
        self.timeUpdate_Level2analysis = now_counter
        self.timeUpdate_recalcLevel2cumulDelta = now_counter
        self.timeUpdate_recalcLevel2BAsize = now_counter
        self.timeUpdate_recalcLevel2BAimbalance = now_counter
        self.timeUpdate_OrderBook = now_counter

        # ----- graphic references and data that should be visualized
        #t = np.linspace(0., 1000, 1001)
        t = np.linspace(now - 6*30*24*3600, now, 100)  # Plot random values with timestamps in the last 6 months
        # history + MarketData updates 250ms: time, open, close, min, max
        self.index_slowUpdate_Limit = None  # index limit between slow and fast updates
        self.ref_slowCandlesVline = None
        self.ref_slowCandlesHline = None
        self.ref_slowCandlesHlineLast = None
        self.ref_slowVolumeVline = None
        self.ref_slowCandles = None
        self.ref_slowCandlesUpdates = None
        self.ref_slowCandles_viewBox = None  # reference for additional viewbox for bars
        self.ref_slowCandles_rescale = None  # reference for invisible plot
        self.data_slowCandles = np.transpose([t,
                                              15 + 3 * np.abs(np.sin(t)),
                                              15 + 3 * np.abs(np.sin(t + 1.5)),
                                              15 - 6 * np.abs(np.sin(t)),
                                              15 + 6 * np.abs(np.sin(t))])
        self.ref_slowCurves = None
        self.data_slowCurveEMA = np.transpose([t,
                                              15 + 1 * np.abs(np.cos(t))])
        self.data_slowCurveATR = np.transpose([t,
                                              15 + 2 * np.abs(np.cos(t)),
                                              15 + 4 * np.abs(np.sin(t + 1.5))])
        self.ref_slowBars = None
        self.ref_slowBarsUpdates = None
        self.ref_slowBars_viewBox = None  # reference for additional viewbox for bars
        self.ref_slowBars_rescale = None  # reference for invisible plot
        self.data_slowBars = np.transpose([t, np.abs(np.sin(t))])
        self.ref_slowCurve = None  # references to viebox + plot/curve itself slow update
        self.ref_slowCurveUpdates = None
        self.data_slowCurve = np.transpose([t, np.abs(np.sin(t))])

        self.data_1min_HighLows = np.zeros((2, 4))  # history + 5s? updates: time, max, min

        self.index_fastUpdate_Limit = None  # index limit between slow and fast updates
        self.ref_fastCandlesVline = None
        self.ref_fastCandlesHline = None
        self.ref_fastVolumeVline = None
        self.ref_fastVolumeHline = None
        # history + MarketData updates 250ms: time, open, close, min, max
        self.ref_fastCandles = None
        self.ref_fastCandlesUpdates = None
        self.data_fastCandles = np.transpose([t,
                                              15 + 3 * np.abs(np.sin(t)),
                                              15 + 3 * np.abs(np.sin(t + 1.5)),
                                              15 - 6 * np.abs(np.sin(t)),
                                              15 + 6 * np.abs(np.sin(t))])
        self.ref_fastCurves = None
        self.data_fastCurveEMA = np.transpose([t,
                                              15 + 1 * np.abs(np.cos(t))])
        self.data_fastCurveATR = np.transpose([t,
                                              15 + 2 * np.abs(np.cos(t)),
                                              15 + 4 * np.abs(np.sin(t + 1.5))])
        # MarketData + updates 250ms: time, bid, ask
        self.ref_fastCurvesBA = []  # reference to bid + ask
        self.data_fastCurveBid = np.transpose([t, np.abs(np.sin(t))])
        self.data_fastCurveAsk = np.transpose([t, np.abs(np.sin(t + 1.5))])
        # history + MarketData updates 250ms: time, volume
        self.ref_fastBars = None
        self.ref_fastBarsUpdates = None
        self.ref_fastBars_viewBox = None  # reference for additional viewbox for bars
        self.ref_fastBars_rescale = None  # reference for invisible plot
        self.data_fastBars = np.transpose([t, np.abs(np.sin(t))])
        # MarketData + updates 250ms: time, trades number
        self.ref_fastCurve = []  # reference to viebox and trades number
        self.data_fastCurve = np.transpose([t, np.abs(np.sin(t))])

        # price, upSize, downSize
        self.ref_VolumeProfVline = None
        self.ref_VolumeProfHline = None
        self.ref_volumeProfBars = None  # references to bars
        self.ref_volumeProfBars_viewBox = None  # reference for additional viewbox for bars
        self.ref_volumeProfBars_rescale = None  # references for invisible plots
        price = 10 + np.arange(1000) / 100
        self.data_volumeProfBuy = np.transpose([price, np.abs(np.sin(price))])
        self.data_volumeProfSell = np.transpose([price, -np.abs(np.sin(price))])
        self.data_volumeProfSum = np.transpose([price, np.abs(np.sin(price))])
        self.data_volumeProf_time_cut_start_index = 0  # index that defines current start index in trades_ticks data
        self.data_volumeProf_update_size = 0  # number of in_Ticks array that are to include in the update

        # custom/cumulative MarketDepth: 2d array
        self.ref_level2Image = None
        self.ref_level2Image_percent = None
        self.ref_level2ImageVline = None
        self.ref_level2ImageHline = None
        self.ref_level2PositionHline = None
        self.data_level2Image_ticks = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, int)  # local storage of raw data
        self.data_level2Image_update = np.array([])  # local storage of pre-sampled data
        self.data_level2Image = np.zeros(LEVEL2_IMAGE_SHAPE, float)   # pre-plot image
        self.data_level2Image_PriceScale = np.array([])
        self.data_level2Image_TimeScale = np.array([])
        self.data_level2Image_TimeScale_ticks = np.array([])
        self.data_level2Image_xyz = [0., 0., 0.]  # data for cross-hair time-price-orders
        # TickTick Last trades
        self.ref_level2Scatter = None
        self.data_level2Scatter_ticks = np.array([])
        self.data_level2Scatter_max = None
        self.data_level2Scatter_update_size = 0  # number of in_Ticks array that are to include in the update
        self.data_level2Scatter = np.transpose([t,
                                                15 + 2 * np.abs(np.sin(t / 100)),  # price
                                                10 * np.abs(np.sin(t / 100))])  # trade size
        # TickTick BA
        self.ref_level2BACurves = None
        self.data_level2BACurves_ticks = np.array([])  # time, bid, ask, bisize, asksize: later two are used for analytical
        self.data_level2BACurves = np.transpose([t,
                                                 15 - 2 * np.abs(np.sin(t / 100)),  #bid price
                                                 15 + 2 * np.abs(np.sin(t / 100 + 1.5))])  #ask price
        # TickTick Analysis
        self.ref_level2AnCurves = []  # references for curve 1+viebox curve 2 and curve 2 itself
        self.data_level2imbalanceSumSizeBACurve_ticks = np.array([])  # sumBid, sumAsk
        self.data_level2imbalanceSumCashBACurve_ticks = np.array([])  # sumBid, sumAsk
        self.data_level2imbalanceAvrgPriceBACurve_ticks = np.array([])  # averageBid, averageAsk
        self.data_level2imbalanceForceBACurve_ticks = np.array([])  # [time, forceBid, forceAsk]
        self.data_level2cumulDelta = np.transpose([t,
                                                    np.abs(np.sin(t / 100))])  # cumulative price up/down
        self.data_level2sizeBACurves = np.transpose([t,
                                                    np.abs(np.sin(t / 100)),  # Bid size or its change in %
                                                    np.abs(np.cos(t / 100))])  # Ask size or its change in %
        self.data_level2imbalanceBACurve = np.transpose([t, np.abs(np.cos(t / 100))])  # imbalance or its change in %

        # Market Depth: (custom local msec time for ticks), volume bid, volume ask, price bid, price ask
        self.ref_orderBookHline = None
        self.ref_orderBookHlineLast = None
        self.ref_orderBookHlineLast_P1per = None
        self.ref_orderBookHlineLast_M1per = None
        self.ref_orderBookHlineLast_P3per = None
        self.ref_orderBookHlineLast_M3per = None
        self.ref_orderBookHlineBidAsk = None
        self.ref_orderBookBars = []  # references for bars
        self.ref_orderBookCurves = []  # references for viebox curves and curves
        self.ref_orderBookBars_viewBox = None  # reference for additional viewbox for bars
        self.ref_orderBookBars_rescale = None  # references for invisible plots

        bid_price = np.transpose(10 + np.arange(500) / 100)
        ask_price = np.transpose(15 + np.arange(500) / 100)
        self.data_orderBookBidBars = np.vstack((bid_price, np.abs(np.sin(bid_price)))).transpose()  # bid price/size
        self.data_orderBookAskBars = np.vstack((ask_price, np.abs(np.sin(ask_price)))).transpose()  # ask price/size
        self.data_orderBookBars_y = 0.  # price data for cross-hair
        self.data_orderBook_update_size = 0

        # assigning/plotting data to the initial dummy plots
        self.init_slow_price_plot(self.slowPrice_Plot)
        self.init_slow_volume_plot(self.slowVolume_Plot)
        self.init_fast_price_plot(self.fastPrice_Plot)
        self.init_fast_volume_plot(self.fastVolume_Plot)
        self.init_volume_profile_plot(self.volumeProfile_Plot)
        self.init_map_level2_plot(self.mapLevel2_Plot)
        self.init_analysis_level2_plot(self.analysisLevel2_Plot)
        self.init_order_book_plot(self.orderBook_Plot)

        self.init_gui_connections()  # setting gui connections and corresponding slot methods
        self.init_gui_misc_elements()  # setting default features of buttons/check boxes/radio buttons/indicators

        logging.debug("Gui window Initialized")

        self.ProcessQueues()

    def addtoLogger(self, msg: str, bkg_color, text_color):
        """Adds log info to LogWidget"""
        self.orderTape_tableWidget.setSortingEnabled(False)

        _row_count = self.orderTape_tableWidget.rowCount()
        self.orderTape_tableWidget.setRowCount(_row_count+1)
        _item = QtWidgets.QTableWidgetItem()
        _item.setData(QtCore.Qt.EditRole, msg)
        _item.setBackground(QtGui.QColor(bkg_color))
        _item.setForeground(QtGui.QColor(text_color))
        self.orderTape_tableWidget.setItem(_row_count, 0, _item)
        self.orderTape_tableWidget.scrollToBottom()

        logging.info(msg)

    def flushQ(self, indexQ):

        while True:
            try:
                q = self.inL1Q.get_nowait() if indexQ == 1 else self.inL2Q.get_nowait()
            except queue.Empty:
                logger.info("Queue flushed")
                break

    def ProcessQueues(self):
        """ main processing loop"""
        #while not self.exitReq:
        _t1 = time.perf_counter()
        self.update_CounterGuiValues()
        self.readQ()
        self.readL1Q()
        self.readL2Q()
        self.readMLQ()
        _t2 = time.perf_counter()
        self.fps_GUIEvents_doubleSpinBox.setValue(1 / (_t2 - _t1))
        self.fps_queue_lineEdit.setText(f"{self.ibQueue_size} => {self.inQ.qsize()}-{self.inL1Q.qsize()}-{self.inL2Q.qsize()}-{self.inMLQ.qsize()}")

    def readQ(self):
        """Read Queue (inQ) method and call corresponding handler method"""
        try:
            q = self.inQ.get(timeout=GUI_QUEUE_RATE)
            #q = self.inQ.get_nowait()
        except queue.Empty:
            pass
        else:
            request = q[0]
            if request == "ERROR":
                self.msg_Error(q)
            elif request == "CONNECTED":
                self.msg_Connected()
            elif request == "CONNECTION_ERROR":
                self.msg_ConnectionError(q)
            elif request == "ACCOUNT_UPDATE":
                self.msg_AccountUpdate(q)
            elif request == "PORTFOLIO_UPDATE":
                self.msg_PortfolioUpdate(q)
            elif request == "POSITION_UPDATE":
                self.msg_PositionUpdate(q)
            elif request == "ACC_PORT_UPDATE_STOPPED":
                self.msg_AccPortStoped()  # includes account updates
            elif request == "COMPANY_INFO_END":
                self.msg_CompanyInfoEnd(q)
            elif request == "NEWS":
                self.msg_NewsUpdate(q)
            elif request == "ORDER":
                self.msg_OrderInfo(q)
            elif request == "IB_QUEUE_SIZE":
                self.ibQueue_size = q[1]

    def readL1Q(self):
        """Read Queue (inL1Q) method and call corresponding handler method"""
        try:
            q = self.inL1Q.get(timeout=GUI_QUEUE_RATE)
            #q = self.inL1Q.get_nowait()
        except queue.Empty:
            pass
        else:
            request = q[0]
            #print("queue1 -> " + datetime.datetime.utcnow().strftime("%M:%S:%f"))
            if request == "SLOW_HISTORY_END":
                self.msg_SlowHistoryEnd(q)
            elif request == "FAST_HISTORY_END":
                self.msg_FastHistoryEnd(q)
            elif request == "SLOW_HISTORY_UPDATE":
                self.msg_SlowHistoryUpdate(q)
            elif request == "HIGHLOW_UPDATE":
                self.msg_HighLowUpdate(q)
            elif request == "FAST_HISTORY_UPDATE":
                self.msg_FastHistoryUpdate(q)
            elif request == "LEVEL1_UPDATE":
                self.msg_Level1Update(q)
            elif request == "SCANNER":
                self.msg_ScannerUpdate(q, isFull=False)
            elif request == "SCANNER_FULL":
                self.msg_ScannerUpdate(q, isFull=True)

    def readL2Q(self):
        """Read Queue (inL2Q) method and call corresponding handler method"""
        try:
            q = self.inL2Q.get(timeout=GUI_QUEUE_RATE)
            #q = self.inL2Q.get_nowait()
        except queue.Empty:
            pass
        else:
            request = q[0]
            #print("queue2 -> " + datetime.datetime.utcnow().strftime("%M:%S:%f"))
            if request == "LEVEL2_UPDATE":
                self.msg_Level2Update(q)

    def readMLQ(self):
        """Read Queue (inQ) method and call corresponding handler method"""
        try:
            q = self.inMLQ.get(timeout=GUI_QUEUE_RATE)
            #q = self.inQ.get_nowait()
        except queue.Empty:
            pass
        else:
            request = q[0]
            if request == "ERROR":
                self.msg_Error(q)
            elif request == "UPDATE":
                self.msg_ML_update(q)

    def trigger_trade(self):

        self.auto_trade(True)

    def auto_trade(self, triggered=False):

        isBuy = self.ml_probaC_spinBox.value() > 50 and self.positionSize_spinBox.value() == 0
        isSell = self.ml_probaC_spinBox.value() < 50 and self.positionSize_spinBox.value() > 0

        isAggressive = self.autotrade_mode_horizontalSlider.value() == 3
        isSafe = self.autotrade_mode_horizontalSlider.value() == 1
        buy_fraction = 0.1 if self.autotrade_buysize_horizontalSlider.value() == 1 else 0.33 if self.autotrade_buysize_horizontalSlider.value() == 2 else 0.5

        buyPrice = self.last_Ask if isSafe else self.last_Bid if isAggressive else (self.last_Bid + self.last_Ask)/2
        sellPrice = self.last_Bid if isSafe else self.last_Ask if isAggressive else (self.last_Bid + self.last_Ask)/2

        a_profit_size = self.autotrade_profitsize_horizontalSlider.value()
        if a_profit_size == 0:
            take_profit = 0.0
        elif a_profit_size == 1:
            take_profit = self.range1minCper_doubleSpinBox.value() * 0.2
        elif a_profit_size == 2:
            take_profit = self.range1minCper_doubleSpinBox.value() * 0.3
        elif a_profit_size == 3:
            take_profit = self.range1minCper_doubleSpinBox.value() * 0.5
        else:
            take_profit = self.range1minCper_doubleSpinBox.value()  # in percents

        if self.autotrade_PL_horizontalSlider.value() == 1:
            stop_loss = take_profit
        elif self.autotrade_PL_horizontalSlider.value() == 2:
            stop_loss = take_profit * 2
        elif self.autotrade_PL_horizontalSlider.value() == 3:
            stop_loss = take_profit * 3
        else:
            stop_loss = take_profit * 10

        if (isBuy and self.ticker_profitper_doubleSpinBox.value() > 0) or triggered:
            self.onBuyLimitPositionsButtonClick(buyPrice, buy_fraction, take_profit, stop_loss)
            return True
        elif isSell:
            self.onCloseLimitPositionsButtonClick(sellPrice, AUTOTRADE_SELL_FRACTION)
            return True
        else:
            return False

    def init_gui_connections(self):
        """assign connection between gui triggers and corresponding methods"""
        # buttons clicked
        self.ibap_Connect_Button.clicked.connect(self.onConnectButtonClick)
        self.ibap_Portfolio_Button.clicked.connect(self.onAccPortButtonClick)
        self.find_Ticker_Button.clicked.connect(lambda: self.onFindButtonClick())
        self.find_positionTicker_Button.clicked.connect(self.onPositionTickerButtonClick)
        self.scannerList_tableWidget.itemDoubleClicked.connect(self.onScanTickerDoubleClick)
        self.ibapi_triggerL1Data_Button.clicked.connect(lambda: self.onLevel1ButtonClick())
        self.ibapi_triggerL2Data_Button.clicked.connect(lambda: self.onLevel2ButtonClick())
        self.streamLevel2NYSE_Button.clicked.connect(self.onLevel2NYSEButtonClick)

        self.getScan_Button.clicked.connect(self.onScannerButtonClick)
        self.getNews_Button.clicked.connect(self.onNewsButtonClick)
        self.slowPrice_OnOff_Button.clicked.connect(self.onSlowPriceVolumeButtonClick)
        self.slowVolume_OnOff_Button.clicked.connect(self.onSlowPriceVolumeButtonClick)
        self.fastPrice_OnOff_Button.clicked.connect(self.onFastPriceVolumeButtonClick)
        self.fastVolume_OnOff_Button.clicked.connect(self.onFastPriceVolumeButtonClick)
        self.plotUpdates_StartStop_Button.clicked.connect(self.onPlotUpdatesStartStopButtonClick)

        self.buy_Button.clicked.connect(self.update_OrderGuiValues)
        self.sell_Button.clicked.connect(self.update_OrderGuiValues)
        self.buyBracket_Button.clicked.connect(self.update_OrderGuiValues)

        self.size100p_Order_Button.clicked.connect(self.update_OrderGuiValues)
        self.size50p_Order_Button.clicked.connect(self.update_OrderGuiValues)
        self.size33p_Order_Button.clicked.connect(self.update_OrderGuiValues)
        self.size20p_Order_Button.clicked.connect(self.update_OrderGuiValues)
        self.size10p_Order_Button.clicked.connect(self.update_OrderGuiValues)
        self.sizeManual_Order_Button.clicked.connect(self.update_OrderGuiValues)

        self.bidPrice_Button.clicked.connect(self.update_OrderGuiValues)
        self.midPrice_Button.clicked.connect(self.update_OrderGuiValues)
        self.askPrice_Button.clicked.connect(self.update_OrderGuiValues)
        self.lastPrice_Button.clicked.connect(self.update_OrderGuiValues)
        self.flatPrice_Button.clicked.connect(self.update_OrderGuiValues)
        self.trackPrice_Button.clicked.connect(self.onTrackPriceOBButtonClick)
        self.manualPrice_Button.clicked.connect(self.update_OrderGuiValues)

        self.placeOrder_Button.clicked.connect(self.onPlaceOrderButtonClick)
        self.cancelOrder_Button.clicked.connect(self.onCancelOrderButtonClick)
        self.stopLossFlat_Button.clicked.connect(self.onStopLossFlatButtonClick)

        self.trigger_autotrade_Button.clicked.connect(self.trigger_trade)

        self.closePositions_Button.clicked.connect(self.onClosePositionsButtonClick)

        self.closeLimit100p_bid_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Bid, 1))
        self.closeLimit50p_bid_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Bid, 0.5))
        self.closeLimit33p_bid_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Bid, 0.33))
        self.closeLimit10p_bid_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Bid, 0.1))

        self.closeLimit100p_ask_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Ask, 1))
        self.closeLimit50p_ask_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Ask, 0.5))
        self.closeLimit33p_ask_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Ask, 0.33))
        self.closeLimit10p_ask_Button.clicked.connect(lambda: self.onCloseLimitPositionsButtonClick(self.last_Ask, 0.1))

        self.buyLimit100p_bid_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Bid, 1))
        self.buyLimit50p_bid_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Bid, 0.5))
        self.buyLimit33p_bid_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Bid, 0.33))
        self.buyLimit10p_bid_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Bid, 0.1))

        self.buyLimit100p_ask_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Ask, 1))
        self.buyLimit50p_ask_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Ask, 0.5))
        self.buyLimit33p_ask_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Ask, 0.33))
        self.buyLimit10p_ask_Button.clicked.connect(lambda: self.onBuyLimitPositionsButtonClick(self.last_Ask, 0.1))

        # combo-boxes changed
        self.slow_timeFrame_comboBox.currentIndexChanged.connect(self.onSlowTimeFrameComboChange)
        self.fast_timeFrame_comboBox.currentIndexChanged.connect(self.onFastTimeFrameComboChange)
        self.span_VolumeProfile_comboBox.currentIndexChanged.connect(self.onSpanVolumeProfileComboChange)
        self.updateRate_MarketDepth_comboBox.currentIndexChanged.connect(self.onMarketDepthUpdateRateComboChange)
        self.imbalanceMode_MarketDepth_comboBox.currentIndexChanged.connect(self.onMarketDepthImbalanceModeComboChange)

        self.logMap_checkBox.stateChanged.connect(self.onLogMapCheckBoxChange)
        self.sumVolProf_checkBox.stateChanged.connect(self.onSumVolProf_CheckBoxChange)
        self.logOrderBook_checkBox.stateChanged.connect(self.onLogOrderBookBoxChange)
        self.isSlowHA_checkBox.stateChanged.connect(self.onSlowHABoxChange)
        self.isFastHA_checkBox.stateChanged.connect(self.onFastHABoxChange)
        self.absCumul_checkBox.stateChanged.connect(self.onAbsCumulBoxChange)

        self.colorScale_horizontalSlider.valueChanged.connect(self.onLogMapSliderChange)
        self.tradesSize_verticalSlider.valueChanged.connect(self.onTradeSizeSliderChange)
        self.cumulSpan_verticalSlider.valueChanged.connect(self.onCumulSpanSliderChange)

        self.timer.timeout.connect(self.ProcessQueues)
        self.timer.start()

    def init_gui_misc_elements(self):
        """makes initial style adjustments"""

        self.buy_Button.setStyleSheet("QPushButton {background-color: %s;} "
                                      "QPushButton::checked {background-color: %s}"
                                      % (GREEN_BUTTON, SELECT_BUTTON))
        self.buyBracket_Button.setStyleSheet("QPushButton {background-color: %s;} "
                                      "QPushButton::checked {background-color: %s}"
                                      % (GREEN_BUTTON, SELECT_BUTTON))
        self.sell_Button.setStyleSheet("QPushButton {background-color: %s;} "
                                       "QPushButton::checked {background-color: %s}"
                                       % (RED_BUTTON, SELECT_BUTTON))
        self.app.setStyleSheet("QPushButton::pressed {background-color: %s}"
                               "QPushButton::checked {background-color: %s}"
                                % (GREEN_BUTTON, SELECT_BUTTON))
        self.scannerList_tableWidget.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

    def clear_PortfolioGuiValue(self):
        # updates quickly open position:

        self.positionPL_doubleSpinBox.setValue(0)
        self.positionPLper_doubleSpinBox.setValue(0)
        self.positionPLperTotal_doubleSpinBox.setValue(0)
        self.averagePrice_doubleSpinBox.setValue(0)
        self.ref_level2PositionHline.setPos(0)
        self.positionSize_spinBox.setValue(0)
        self.positionValue_doubleSpinBox.setValue(0)
        self.positionValuePer_doubleSpinBox.setValue(0)
        self.totalValue_doubleSpinBox.setValue(0)
        self.totalCash_doubleSpinBox.setValue(0)

        self.dayPL_doubleSpinBox.setValue(0)
        self.dayPLper_doubleSpinBox.setValue(0)

        self.positionTicker_lineEdit.setText("")

    def update_PositionGuiValue(self):
        # updates quickly open position:

        self.positionValue_doubleSpinBox.setValue(self.last_Price * self.positionSize_spinBox.value())
        if self.totalValue_doubleSpinBox.value() > 0.0:
            self.positionValuePer_doubleSpinBox.setValue(
                100 * self.last_Price * self.positionSize_spinBox.value() / self.totalValue_doubleSpinBox.value())
        _PL = self.positionValue_doubleSpinBox.value() - self.averagePrice_doubleSpinBox.value()*self.positionSize_spinBox.value()
        _posPLper = 100. * (
                    self.last_Price / self.averagePrice_doubleSpinBox.value() - 1) if self.averagePrice_doubleSpinBox.value() != 0 else 0
        _posPLperTotal = _posPLper*self.positionValuePer_doubleSpinBox.value()/100

        self.positionPL_doubleSpinBox.setValue(_PL)
        self.positionPL_doubleSpinBox.setStyleSheet("background-color: %s}" % (
            DRED if _posPLper < 0 else DGREY if _posPLper < 1 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
        self.positionPLper_doubleSpinBox.setValue(_posPLper)
        self.positionPLper_doubleSpinBox.setStyleSheet("background-color: %s}" % (
            DRED if _posPLper < 0 else DGREY if _posPLper < 1 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
        self.positionPLperTotal_doubleSpinBox.setValue(_posPLperTotal)
        self.positionPLperTotal_doubleSpinBox.setStyleSheet("background-color: %s}" % (
            DRED if _posPLperTotal < 0 else DGREY if _posPLperTotal < 1 else DGREEN if _posPLperTotal < 5 else DBLUE if _posPLperTotal < 10 else PURPLE))

    def update_OrderGuiValues(self):
        """Updates Gui values for the ordering values"""
        _p = CAPITAL_FRACTION  # fraction of cash that can be used
        if self.bidPrice_Button.isChecked():
            _target_price = self.last_Bid
        elif self.askPrice_Button.isChecked():
            _target_price = self.last_Ask
        elif self.midPrice_Button.isChecked():
            _target_price = self.last_MidPoint
        elif self.lastPrice_Button.isChecked():
            _target_price = self.last_Price

        elif self.flatPrice_Button.isChecked():
            _target_price = self.averagePrice_doubleSpinBox.value()

        elif self.trackPrice_Button.isChecked():
            _target_price = self.armedPrice_doubleSpinBox.value()
        elif self.manualPrice_Button.isChecked():
            _target_price = self.armedPrice_doubleSpinBox.value()
        else:
            _target_price = self.last_Price

        self.armedPrice_doubleSpinBox.setValue(_target_price)

        if self.buy_Button.isChecked() or self.buyBracket_Button.isChecked():  # buy OR bracket: should not be possible to borrow money
            _limit_size = int(_p * self.last_Cash // max(self.last_Ask,
                                                    self.armedPrice_doubleSpinBox.value())) if self.last_Ask > 0 or self.armedPrice_doubleSpinBox.value() > 0 else 0  # double safe
            # _limit_price = not relevant when buying

            try:
                if self.size100p_Order_Button.isChecked():
                    _new_size = int((_p * self.last_Cash / 1) // _target_price)
                elif self.size50p_Order_Button.isChecked():
                    _new_size = int((_p * self.last_Cash / 2) // _target_price)
                elif self.size33p_Order_Button.isChecked():
                    _new_size = int((_p * self.last_Cash / 3) // _target_price)
                elif self.size20p_Order_Button.isChecked():
                    _new_size = int((_p * self.last_Cash / 5) // _target_price)
                elif self.size10p_Order_Button.isChecked():
                    _new_size = int((_p * self.last_Cash / 10) // _target_price)
                elif self.sizeManual_Order_Button.isChecked():
                    _new_size = self.armedShares_spinBox.value()
                else:
                    _new_size = _limit_size
                _new_size = min(_new_size, _limit_size)
                self.armedShares_spinBox.setValue(_new_size)
            except: pass

        elif self.sell_Button.isChecked():  # should not be possible to short
            _limit_size = self.last_PositionSize
            #_limit_price = not relevant for selling
            if self.size100p_Order_Button.isChecked():
                _new_size = int(_limit_size // 1)
            elif self.size50p_Order_Button.isChecked():
                _new_size = int(_limit_size // 2)
            elif self.size33p_Order_Button.isChecked():
                _new_size = int(_limit_size // 3)
            elif self.size20p_Order_Button.isChecked():
                _new_size = int(_limit_size // 5)
            elif self.size10p_Order_Button.isChecked():
                _new_size = int(_limit_size // 10)
            elif self.sizeManual_Order_Button.isChecked():
                _new_size = min(_limit_size, self.armedShares_spinBox.value())
            else:
                _new_size = _limit_size

            self.armedShares_spinBox.setValue(_new_size)

    def update_CounterGuiValues(self):
        # updates 1 and 5 min counters:

        _time = int(time.time())

        self.min1Counter_spinBox.setValue(60-_time % 60)
        self.min5Counter_spinBox.setValue(300 - _time % 300)

        if (60-_time % 60) == 1:
            if self.enable_autotrade_Button.isChecked() and not self.autotraded:
                self.autotraded = self.auto_trade()
        else:
            self.autotraded = False

    def update_HighLow_Values(self):
        self.high5minP_doubleSpinBox.setValue(self.data_1min_HighLows[0, 0])
        _high5minC = max(self.data_1min_HighLows[0, 1], self.last_Price)
        self.high5minC_doubleSpinBox.setValue(_high5minC)
        self.high1minP_doubleSpinBox.setValue(self.data_1min_HighLows[0, 2])
        _high1minC = max(self.data_1min_HighLows[0, 3], self.last_Price)
        self.high1minC_doubleSpinBox.setValue(_high1minC)

        self.low5minP_doubleSpinBox.setValue(self.data_1min_HighLows[1, 0])
        _low5minC = min(self.data_1min_HighLows[1, 1], self.last_Price)
        self.low5minC_doubleSpinBox.setValue(_low5minC)
        self.low1minP_doubleSpinBox.setValue(self.data_1min_HighLows[1, 2])
        _low1minC = min(self.data_1min_HighLows[1, 3], self.last_Price)
        self.low1minC_doubleSpinBox.setValue(_low1minC)

        self.range5minPper_doubleSpinBox.setValue(
            (self.data_1min_HighLows[0, 0] - self.data_1min_HighLows[1, 0]) * 100 / self.last_Price)
        self.range1minPper_doubleSpinBox.setValue(
            (self.data_1min_HighLows[0, 2] - self.data_1min_HighLows[1, 2]) * 100 / self.last_Price)
        self.range5minCper_doubleSpinBox.setValue(
            (_high5minC - _low5minC) * 100 / self.last_Price)
        self.range1minCper_doubleSpinBox.setValue(
            (_high1minC - _low1minC) * 100 / self.last_Price)

        self.higherHigh_indicator_lineEdit.setStyleSheet("background-color: %s}" % (
            BLACK if self.high1minC_doubleSpinBox.value() < self.high1minP_doubleSpinBox.value() else DGREEN))
        self.lowerLow_indicator_lineEdit.setStyleSheet("background-color: %s}" % (
            DRED if self.low1minC_doubleSpinBox.value() < self.low1minP_doubleSpinBox.value() else BLACK))

        self.higherHigh5_indicator_lineEdit.setStyleSheet("background-color: %s}" % (
            BLACK if self.high5minC_doubleSpinBox.value() < self.high5minP_doubleSpinBox.value() else DGREEN))
        self.lowerLow5_indicator_lineEdit.setStyleSheet("background-color: %s}" % (
            DRED if self.low5minC_doubleSpinBox.value() < self.low5minP_doubleSpinBox.value() else BLACK))

        a_profit_size = self.autotrade_profitsize_horizontalSlider.value()
        if a_profit_size == 0:
            take_profit = 0.0
        elif a_profit_size == 1:
            take_profit = max(self.range1minCper_doubleSpinBox.value(), self.range1minPper_doubleSpinBox.value()) * 0.2
        elif a_profit_size == 2:
            take_profit = max(self.range1minCper_doubleSpinBox.value(), self.range1minPper_doubleSpinBox.value()) * 0.3
        elif a_profit_size == 3:
            take_profit = max(self.range1minCper_doubleSpinBox.value(), self.range1minPper_doubleSpinBox.value()) * 0.5
        else:
            take_profit = max(self.range1minCper_doubleSpinBox.value(), self.range1minPper_doubleSpinBox.value())  # in percents

        strategy = self.autotrade_mode_horizontalSlider.value()
        take_profit -= self.ticker_spreadper_doubleSpinBox.value()*(0 if strategy == 3 else 1 if strategy == 1 else 0.5)
        self.ticker_profitper_doubleSpinBox.setValue(take_profit)
        self.ticker_profitper_doubleSpinBox.setStyleSheet("background-color: %s}" % (DRED if take_profit <= 0 else DGREEN))


    def onTrackPriceOBButtonClick(self):
        self.armedPrice_doubleSpinBox.setValue(self.data_orderBookBars_y)
        self.update_OrderGuiValues()

    def onSlowTimeFrameComboChange(self):
        """requests/gets/updates plot of new history based on selection"""
        #self.addtoLogger(self.slow_timeFrame_comboBox.currentText())
        self.onFindButtonClick(comboChange=True)

    def onSlowHABoxChange(self):
        if self.historyValidate("TRADES_SLOW"):
            self.update_slow_price_plot(fullUpdate=True)

    def onFastTimeFrameComboChange(self):
        """requests/gets/updates plot of new history based on selection"""
        #self.addtoLogger(self.fast_timeFrame_comboBox.currentText())
        self.onFindButtonClick(comboChange=True)

    def onFastHABoxChange(self):
        if self.historyValidate("TRADES_FAST"):
            self.update_fast_price_plot(fullUpdate=True)

    def onSpanVolumeProfileComboChange(self):
        self.recalculate_VolumeProfile(fullUpdate=True, isSum=self.sumVolProf_checkBox.isChecked())

    def onSumVolProf_CheckBoxChange(self):
        self.recalculate_VolumeProfile(fullUpdate=True, isSum=self.sumVolProf_checkBox.isChecked())

    def onMarketDepthUpdateRateComboChange(self):
        self.recalculate_Level2_BA(True)
        #self.recalculate_Level2_sizeBA(True)
        self.recalculate_Level2_imbalanceBA(True)
        self.recalculate_Level2_Trades(True)
        self.recalculate_Map(True)

    def onMarketDepthImbalanceModeComboChange(self):
        self.recalculate_Level2_imbalanceBA(True)

    def onLogMapCheckBoxChange(self):
        self.colorScale_horizontalSlider.setValue(0)
        self.recalculate_Map(True)

    def onAbsCumulBoxChange(self):
        self.cumulSpan_verticalSlider.setValue(300 if not self.absCumul_checkBox.isChecked() else 10)
        self.recalculate_Level2_cumulDelta(fullUpdate=True)

    def onLogMapSliderChange(self):
        self.update_map_level2image_plot()

    def onTradeSizeSliderChange(self):
        self.update_map_level2_plot()

    def onCumulSpanSliderChange(self):
        self.recalculate_Level2_cumulDelta(fullUpdate=True)

    def onLogOrderBookBoxChange(self):
        self.update_order_book_plot()

    def onConnectButtonClick(self):
        """start parallel event/readQ loop with passing control to the main QT event handler sometimes"""
        #playsound("sounds/button.wav")
        if not self.isConnected:
            self.ibapi_Status_lineEdit.setText("...")
            logging.info("Connecting to TWS")
            #self.addtoLogger("Connecting to TWS")
            self.outQ.put_nowait(("CONNECT",
                                  self.ibapi_IP_lineEdit.text(),
                                  int(self.ibapi_Port_spinBox.value()),
                                  int(self.ibapi_ID_spinBox.value())))
        else:
            self.ibap_Connect_Button.setChecked(True)

    def onAccPortButtonClick(self):
        """connect to the IB API via queue"""
        self.clear_PortfolioGuiValue()
        if self.isConnected:
            if self.isAccountPortfolioUpdating:
                logging.info("Account/Portfolio data stop requested")
                #self.addtoLogger("Account/Portfolio data stop requested")
                self.ibap_Portfolio_Button.setChecked(False)
                self.isAccountPortfolioUpdating = False
                self.ibap_Portfolio_Button.setText("Update")
                self.outQ.put_nowait(("ACC_PORT_STOP",))  # last two parameters for future use: number of bars for slow and fast history
            else:
                logging.info("Account/Portfolio data requested")
                #self.addtoLogger("Account/Portfolio data requested")
                self.ibap_Portfolio_Button.setChecked(True)
                self.isAccountPortfolioUpdating = True
                self.ibap_Portfolio_Button.setText("Stop")
                self.outQ.put_nowait(("ACC_PORT",))  # last two parameters for future use: number of bars for slow and fast history
        else:
            self.ibap_Portfolio_Button.setChecked(False)

    def onFindButtonClick(self, comboChange=False, ticker="*"):
        """fin company in the IB API via queue"""
        if self.isConnected:
            logging.info("Collecting history data")
            #self.addtoLogger("Collecting history data")
            if self.isLevel1Updating:
                self.onLevel1ButtonClick(ticker)
                time.sleep(0.5)  # to be sure that history and streams stopped
                self.flushQ(1)
                self.onLevel1ButtonClick(ticker)
            else:
                # clear initial arrays
                self.data_slowCandles = np.array([])  # [time, open, close, low, high]
                self.data_slowBars = np.array([])  # [time, volume]
                self.data_slowCurve = np.array([])  # [time, number of trades]
                self.data_fastCandles = np.array([])  # [time, open, close, low, high]
                self.data_fastBars = np.array([])  # history + 5s? updates: time, volume
                self.data_fastCurve = np.array([])  # history + 5s? updates: time, number of trades
                self.data_fastCurveBid = np.array([])  # history + 5s? updates: time, bid
                self.data_fastCurveAsk = np.array([])  # history + 5s? updates: time, ask

                self.ticker_Status_lineEdit.setText("requesting company data...")
                self.outQ.put_nowait(("FIND",
                                      ticker if ticker != "*" else self.ticker_Symbol_lineEdit.text(),
                                      self.slow_timeFrame_comboBox.currentText(),
                                      self.fast_timeFrame_comboBox.currentText(),
                                      1 if self.rthMode_Data_comboBox.currentText() == "RTH" else 0,
                                      500, 500))  # last two parameters for future use: number of bars for slow and fast history
            if self.isLevel2Updating and not comboChange:
                self.onLevel2ButtonClick(ticker)
                time.sleep(0.5)  # to be sure that history and streams stopped
                self.flushQ(2)
                self.onLevel2ButtonClick(ticker)
        else:
            self.find_Ticker_Button.setChecked(False)

    def onPositionTickerButtonClick(self):
        self.ticker_Symbol_lineEdit.setText(self.positionTicker_lineEdit.text())
        self.onFindButtonClick(comboChange=False, ticker=self.positionTicker_lineEdit.text())

    def onScanTickerDoubleClick(self):
        _ticker = self.scannerList_tableWidget.currentItem().text()
        self.ticker_Symbol_lineEdit.setText(_ticker)
        self.onFindButtonClick(comboChange=False, ticker=_ticker)

    def onLevel1ButtonClick(self, ticker="*"):
        """connect to the LEVEL1 stream IB API via queue"""
        if self.isConnected:
            if self.isLevel1Updating:
                logging.info("Level1 stream stop requested")
                #self.addtoLogger("Level1 stream stop requested")
                self.ibapi_triggerL1Data_Button.setChecked(False)
                self.isLevel1Updating = False
                self.outQ.put_nowait(("LEVEL1_STOP",))
            else:
                # clear initial arrays
                self.data_slowCandles = np.array([])  # [time, open, close, low, high]
                self.data_slowBars = np.array([])  # [time, volume]
                self.data_slowCurve = np.array([])  # [time, number of trades]

                self.data_fastCandles = np.array([])  # [time, open, close, low, high]
                self.data_fastBars = np.array([])  # history + 5s? updates: time, volume
                self.data_fastCurve = np.array([])  # history + 5s? updates: time, number of trades
                self.data_fastCurveBid = np.array([])  # history + 5s? updates: time, bid
                self.data_fastCurveAsk = np.array([])  # history + 5s? updates: time, ask

                logging.info("Level1 stream requested")
                #self.addtoLogger("Level1 stream requested")
                self.isLevel1Updating = True
                self.ibapi_triggerL1Data_Button.setChecked(True)
                self.outQ.put_nowait(("LEVEL1",
                                      ticker if ticker != "*" else self.ticker_Symbol_lineEdit.text(),
                                      self.slow_timeFrame_comboBox.currentText(),
                                      self.fast_timeFrame_comboBox.currentText(),
                                      1 if self.rthMode_Data_comboBox.currentText() == "RTH" else 0,
                                      500, 500))  # last two parameters for future use: number of bars for slow and fast history
        else:
            self.ibapi_triggerL1Data_Button.setChecked(False)

    def onLevel2ButtonClick(self, ticker="*"):
        """connect to the LEVEL1 stream IB API via queue"""
        if self.isConnected:
            if self.isLevel2Updating:
                self.outQ.put_nowait(("LEVEL2_STOP",))
                logging.info("Level2 stream stop requested")
                #self.addtoLogger("Level2 stream stop requested")
                self.ibapi_triggerL2Data_Button.setChecked(False)
                self.isLevel2Updating = False
            else:
                # clear initial arrays
                # Tick trades
                self.data_level2Scatter_ticks = np.array([])  # [time, price, size]
                self.data_level2Scatter = np.array([])
                self.data_level2Scatter_update_size = 0
                # TickTick BA
                self.data_level2BACurves_ticks = np.array([])  # time, bid, ask, bisize, asksize: later two are used for analytical
                self.data_level2BACurves = np.array([])  # [time, bid price, ask price]
                # TickTick Analysis
                self.data_level2imbalanceSumSizeBACurve_ticks = np.array([])  # [time, sumBid, sumAsk]
                self.data_level2imbalanceSumCashBACurve_ticks = np.array([])  # [time, sumBid, sumAsk]
                self.data_level2imbalanceAvrgPriceBACurve_ticks = np.array([])  # timne, averageBid, averageAsk
                self.data_level2imbalanceForceBACurve_ticks = np.array([])  # [time, forceBid, forceAsk]
                self.data_level2imbalanceBACurve = np.array([])  # [time, imbalance abs/avrgPriced/force]
                self.data_level2cumulDelta = np.array([])
                self.data_level2sizeBACurves = np.array([])  # [time, Bid+Ask size]

                self.data_volumeProfBuy = np.array([])  # [price, Up volume]
                self.data_volumeProfSell = np.array([])  # [price, Down volume]
                self.data_volumeProfSum = np.array([])  # [price, Up+Down volume]
                self.data_volumeProf_update_size = 0
                self.data_orderBookBidBars = np.zeros((MARKET_DEPTH_LEVELS, 2), float)
                self.data_orderBookAskBars = np.zeros((MARKET_DEPTH_LEVELS, 2), float)
                self.data_orderBook_update_size = 0
                # Image
                self.data_level2Image_ticks = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, int)  # local storage of raw data
                self.data_level2Image_update = np.array([])  # local storage of pre-sampled data
                self.data_level2Image = np.zeros(LEVEL2_IMAGE_SHAPE, float)  # pre-plot image
                self.data_level2Image_PriceScale = np.array([])
                self.data_level2Image_TimeScale = np.array([])
                self.data_level2Image_TimeScale_ticks = np.array([])

                logging.info("Level2 stream requested")
                #self.addtoLogger("Level2 stream requested")
                self.isLevel2Updating = True
                self.ibapi_triggerL2Data_Button.setChecked(True)
                self.outQ.put_nowait(
                    ("LEVEL2", ticker if ticker != "*" else self.ticker_Symbol_lineEdit.text(), MARKET_DEPTH_LEVELS))
        else:
            self.ibapi_triggerL2Data_Button.setChecked(False)

    def onLevel2NYSEButtonClick(self):
        self.outQ.put_nowait(("LEVEL2_NYSE", self.streamLevel2NYSE_Button.isChecked()))

    def onPlaceOrderButtonClick(self):
        """Place order to IB API via queue"""
        self.onCancelOrderButtonClick()
        if self.isConnected:
            _symbol = self.ticker_Symbol_lineEdit.text()
            _price = float(self.armedPrice_doubleSpinBox.value())
            _size = int(self.armedShares_spinBox.value())
            _price = float(self.armedPrice_doubleSpinBox.value())

            if self.buyBracket_Button.isChecked():
                _action = "BUY"
                _type = "BKT"
                _isProfit = self.bracketProfitOrder_Button.isChecked()
                _isTrail = self.bracketLossTrailOrder_Button.isChecked()
                _isLossLimit = self.bracketLossLimitOrder_Button.isChecked()
                _profit_per = 0
                _profit_per += 0.25 if self.bracket_profit025per_Button.isChecked() else 0
                _profit_per += 0.5 if self.bracket_profit05per_Button.isChecked() else 0
                _profit_per += 1 if self.bracket_profit1per_Button.isChecked() else 0
                _loss_per = 0
                _loss_per += 0.5 if self.bracket_loss05per_Button.isChecked() else 0
                _loss_per += 1 if self.bracket_loss1per_Button.isChecked() else 0
                _loss_per += 3 if self.bracket_loss3per_Button.isChecked() else 0
                _price_profit = round(_price*(1+0.01*_profit_per), self.priceDecimals)
                _price_loss = round(_price*(1-0.01*_loss_per), self.priceDecimals)
                # symbol, type, action, size, price_buy, price_take profit, price loss abs/per, isProfit,isTrail, isLimit
                q = ("PLACE_ORDER", _symbol, _type, _action, _size,
                     _price, _price_profit, _loss_per if _isTrail else _price_loss,
                     _isProfit, _isTrail, _isLossLimit)
            elif self.buy_Button.isChecked():
                _action = "BUY"
                _type = "LMT" if self.limitOrder_Button.isChecked() else "MKT" if self.marketOrder_Button.isChecked() else "MP" if self.midpointOrder_Button.isChecked() else "STP"
                q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price)  # symbol, type, action, size, price
            else:
                _action = "SELL"
                _type = "LMT" if self.limitOrder_Button.isChecked() else "MKT" if self.marketOrder_Button.isChecked() else "MP" if self.midpointOrder_Button.isChecked() else "STP"
                q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price)  # symbol, type, action, size, price

            if _size > 0:
                self.outQ.put_nowait(q)

                # exporter_l2 = pge.ImageExporter(self.mapLevel2_Plot.scene())
                # exporter_slow = pge.ImageExporter(self.slowPrice_Plot.scene())
                # exporter_l2.parameters()["width"] = 1000
                # exporter_slow.parameters()["width"] = 2000
                # _filename_l2 = time.strftime(
                #     "log/order_snapshots/%y%m%d_%H%M%S_Level2" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                #         _size) + "_" + str(_price) + ".png")
                # _filename_slow = time.strftime(
                #     "log/order_snapshots/%y%m%d_%H%M%S_Candles" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                #         _size) + "_" + str(_price) + ".png")
                # exporter_l2.export(_filename_l2)
                # exporter_slow.export(_filename_slow)
                _filename = time.strftime(
                    "log/order_snapshots/%y%m%d_%H%M%S" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                        _size) + "_" + str(_price) + ".png")

                QtGui.QScreen.grabWindow(self.app.primaryScreen(),
                                         QtWidgets.QApplication.desktop().winId()).save(_filename, 'png')

                logging.info("Order Placed"+str(q))

    def onCancelOrderButtonClick(self):
        """Cancel order to IB API via queue"""
        if self.isConnected:
            if self.buyBracket_Button.isChecked():
                q = ("CANCEL_ORDER", "BKT")
            else:
                q = ("CANCEL_ORDER", "NON_GLOBAL")
            self.outQ.put_nowait(q)
            logging.info("Order Cancel requested")

    def onStopLossFlatButtonClick(self):
        self.onCancelOrderButtonClick()
        if self.isConnected:
            _symbol = self.ticker_Symbol_lineEdit.text()
            _type = "STP"
            _action = "SELL"
            _size = int(self.positionSize_spinBox.value())  # all currently holding
            _price = float(self.averagePrice_doubleSpinBox.value())
            q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price)  # symbol, type, action, size, stop-loss price
            self.outQ.put_nowait(q)

            # exporter_l2 = pge.ImageExporter(self.mapLevel2_Plot.scene())
            # exporter_slow = pge.ImageExporter(self.slowPrice_Plot.scene())
            # exporter_l2.parameters()["width"] = 1000
            # exporter_slow.parameters()["width"] = 2000
            # _filename_l2 = time.strftime(
            #     "log/order_snapshots/%y%m%d_%H%M%S_Level2" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
            #         _size) + "_" + str(_price) + ".png")
            # _filename_slow = time.strftime(
            #     "log/order_snapshots/%y%m%d_%H%M%S_Candles" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
            #         _size) + "_" + str(_price) + ".png")
            # exporter_l2.export(_filename_l2)
            # exporter_slow.export(_filename_slow)
            _filename = time.strftime(
                "log/order_snapshots/%y%m%d_%H%M%S" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                    _size) + "_" + str(_price) + ".png")

            QtGui.QScreen.grabWindow(self.app.primaryScreen(),
                                     QtWidgets.QApplication.desktop().winId()).save(_filename, 'png')

            logging.info("StopLoss Flat requested" + str(q))

    def onClosePositionsButtonClick(self):
        """Close Positions with Market Order to IB API via queue"""
        self.onCancelOrderButtonClick()
        if self.isConnected:
            _symbol = self.ticker_Symbol_lineEdit.text()
            _type = "MKT"
            _action = "SELL"
            _size = int(self.positionSize_spinBox.value())  # all currently holding
            if _size > 0:
                _price = float(self.armedPrice_doubleSpinBox.value())
                q = ("PLACE_ORDER", _symbol, _type, _action, _size)  # symbol, type, action, size
                self.outQ.put_nowait(q)

                _filename = time.strftime(
                    "log/order_snapshots/%y%m%d_%H%M%S" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                        _size) + "_" + str(_price) + ".png")

                QtGui.QScreen.grabWindow(self.app.primaryScreen(),
                                         QtWidgets.QApplication.desktop().winId()).save(_filename, 'png')

                logging.info("Close Position requested" + str(q))


    def onCloseLimitPositionsButtonClick(self, price, fraction):
        """Close Positions with Limit Order to IB API via queue"""
        self.onCancelOrderButtonClick()
        if self.isConnected:
            _symbol = self.ticker_Symbol_lineEdit.text()
            _type = "LMT"
            _action = "SELL"
            _size = int(self.positionSize_spinBox.value()*fraction)  # all currently holding
            if _size > 0:
                _price = round(price, self.priceDecimals)
                q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price)  # symbol, type, action, size
                self.outQ.put_nowait(q)

                _filename = time.strftime(
                    "log/order_snapshots/%y%m%d_%H%M%S" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                        _size) + "_" + str(_price) + ".png")

                QtGui.QScreen.grabWindow(self.app.primaryScreen(),
                                         QtWidgets.QApplication.desktop().winId()).save(_filename, 'png')

                logging.info(f"Close Limit {fraction} of position requested" + str(q))

    def onBuyLimitPositionsButtonClick(self, price, fraction, take_profit=0.0, stop_loss=0.0):
        """Close Positions with Limit Order to IB API via queue"""
        self.onCancelOrderButtonClick()
        if self.isConnected:
            _symbol = self.ticker_Symbol_lineEdit.text()
            _type = "LMT"
            _action = "BUY"
            _price = round(price, self.priceDecimals)
            _limit_size = int(CAPITAL_FRACTION * self.last_Cash // (_price if _price > 0 else 99999))
            _size = int(_limit_size * fraction)
            if _size > 0:
                if take_profit > 0.0:
                    _type = "BKT"
                    _isProfit = True
                    _isTrail = False
                    _isLossLimit = True  # todo limit vs market stop
                    _profit_per = take_profit
                    _loss_per = stop_loss
                    _price_profit = round(_price*(1+0.01*_profit_per), self.priceDecimals)
                    _price_loss = round(_price*(1-0.01*_loss_per), self.priceDecimals)
                    # symbol, type, action, size, price_buy, price_take profit, price loss abs/per, isProfit,isTrail, isLimit
                    q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price,
                         _price_profit, _loss_per if _isTrail else _price_loss,
                         _isProfit, _isTrail, _isLossLimit)
                else:
                    q = ("PLACE_ORDER", _symbol, _type, _action, _size, _price)  # symbol, type, action, size
                self.outQ.put_nowait(q)

                _filename = time.strftime(
                    "log/order_snapshots/%y%m%d_%H%M%S" + "_" + _symbol + "_" + _type + "_" + _action + "_" + str(
                        _size) + "_" + str(_price) + ".png")

                QtGui.QScreen.grabWindow(self.app.primaryScreen(),
                                         QtWidgets.QApplication.desktop().winId()).save(_filename, 'png')

                logging.info(f"Open Limit {fraction} of position requested" + str(q))

    def onScannerButtonClick(self):
        """Start Stop Scanner data processing/updates"""
        if self.isConnected:
            self.scannerList_tableWidget.clearContents()
            self.scannerList_tableWidget.setRowCount(0)
            q = ("SCAN", self.scannerMode_comboBox.currentText(), self.scannerLocation_comboBox.currentText())
            self.outQ.put_nowait(q)
            logging.info("Scanner requested")

    def onNewsButtonClick(self):
        """Start Stop Scanner data processing/updates"""
        if self.isConnected:
            q = ("NEWS", self.ticker_Symbol_lineEdit.text())  # symbol
            self.outQ.put_nowait(q)
            logging.info("Get News requested")

    def onPlotUpdatesStartStopButtonClick(self):
        """Single-shot figures update"""

        if self.historyValidate("TRADES_SLOW"):
            self.update_slow_price_plot()
            self.update_slow_volume_plot()
        if self.historyValidate("TRADES_FAST") and self.historyValidate("BID") and self.historyValidate("ASK"):
            self.update_fast_price_plot()
            self.update_fast_volume_plot()
        self.update_volume_profile_plot()
        self.update_map_level2_plot()
        self.update_analysis_level2_plot()
        self.update_order_book_plot()

    def onSlowPriceVolumeButtonClick(self):
        if self.historyValidate("TRADES_SLOW"):
            self.update_slow_price_plot(autoscale=False, fullUpdate=True)

    def onFastPriceVolumeButtonClick(self):
        if self.historyValidate("TRADES_FAST") and self.historyValidate("BID") and self.historyValidate("ASK"):
            self.update_fast_price_plot(autoscale=False, fullUpdate=True)

    def init_slow_price_plot(self, figure):
        """initialize Slow plot"""
        # set style 
        #figure.setTitle("S-Price")
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("bottom").setStyle(showValues=False)
        figure.getAxis("left").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)

        vb = figure.getViewBox()
        p_rescale = pg.ViewBox()
        figure.scene().addItem(p_rescale)
        p_rescale.setXLink(figure)
        p_rescale.setYLink(figure)

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        hLineLast = pg.InfiniteLine(pen="g", angle=0, movable=False)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        figure.addItem(hLineLast, ignoreBounds=True)
        self.ref_slowCandlesVline = vLine
        self.ref_slowCandlesHline = hLine
        self.ref_slowCandlesHlineLast = hLineLast

        def update_views():
            p_rescale.setGeometry(vb.sceneBoundingRect())
            p_rescale.linkedViewChanged(vb, p_rescale.XYAxes)
        update_views()
        vb.sigResized.connect(update_views)

        # +update/scale
        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                _dt = datetime.datetime.fromtimestamp(int(mousePoint.x())).strftime('%Y-%m-%d %H:%M:%S')
                self.slowValues_lineEdit.setText(f"{_dt}/{mousePoint.y():.2f}")
                #figure.setTitle(f"S-Price (%0.1f/%0.1f)" % (mousePoint.x(), mousePoint.y()))
                vLine.setPos(mousePoint.x())
                self.ref_slowVolumeVline.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())
        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        #   Actual plot section

        _atr = ta.atr(self.data_slowCandles[:, (1, 4, 3, 2)], 10)  # converts to OHLC candles
        _ema = ta.ema(self.data_slowCandles[:, 2], 10)
        self.data_slowCurveATR[:, 1] = _ema - 2 * _atr
        self.data_slowCurveATR[:, 2] = _ema + 2 * _atr

        atr_down = pg.PlotDataItem(self.data_slowCurveATR[:, 0],
                                   self.data_slowCurveATR[:, 1],
                                   pen=pg.mkPen(YELLOW, width=2))
        atr_up = pg.PlotDataItem(self.data_slowCurveATR[:, 0],
                                 self.data_slowCurveATR[:, 2],
                                 pen=pg.mkPen(BLUE, width=2))
        ema = pg.PlotDataItem(self.data_slowCurveEMA[:, 0],
                                 self.data_slowCurveEMA[:, 1],
                                 pen=pg.mkPen(DGREEN, width=2))
        self.ref_slowCurves = [atr_down, atr_up, ema]
        figure.addItem(self.ref_slowCurves[0])
        figure.addItem(self.ref_slowCurves[1])
        figure.addItem(self.ref_slowCurves[2])

        self.index_slowUpdate_Limit = max(self.data_slowCandles.shape[0] - CANDLES_UPDATE_LIMIT, 3)
        self.ref_slowCandles = CandlestickItem(self.data_slowCandles[:self.index_slowUpdate_Limit],
                                                   self.isSlowHA_checkBox.isChecked())
        self.ref_slowCandlesUpdates = CandlestickItem(self.data_slowCandles[self.index_slowUpdate_Limit-1:, :],
                                                      self.isSlowHA_checkBox.isChecked())
        self.ref_slowCandles_rescale = pg.PlotDataItem(self.data_slowCandles[:, 0], self.data_slowCandles[:, 2], pen=RED)

        figure.addItem(self.ref_slowCandles_rescale)
        p_rescale.addItem(self.ref_slowCandles)
        p_rescale.addItem(self.ref_slowCandlesUpdates)
        self.ref_slowCandles_viewBox = p_rescale

        self.ref_slowCandlesHlineLast.setPos(self.data_slowCandles[-1, 2])

        vb.setAspectLocked(lock=False)
        vb.setMouseEnabled(y=False)
        vb.setAutoVisible(y=True)
        vb.enableAutoRange(axis='y', enable=True)

        figure.setXLink(self.slowVolume_Plot)

    def update_slow_price_plot(self, autoscale=False, fullUpdate=False):
        """updates Slow plot: clear/plot new data"""
        start = time.perf_counter()

        if self.slowPrice_OnOff_Button.isChecked() and abs(start-self.timeUpdate_SlowCandle) > 1/FRAME_RATE_GUI:

            vb = self.slowPrice_Plot.getViewBox()
            [[xmin, _], [_, _]] = vb.viewRange()
            vb.setXRange(min=xmin, max=1.05*self.data_slowCandles[-1, 0]-0.05*xmin, padding=0)

            if fullUpdate or self.index_slowUpdate_Limit < self.data_slowCandles.shape[0] - 2 * CANDLES_UPDATE_LIMIT:  # change index limit
                self.index_slowUpdate_Limit = max(self.data_slowCandles.shape[0] - CANDLES_UPDATE_LIMIT, 3)
                self.ref_slowCandles.setData(self.data_slowCandles[:self.index_slowUpdate_Limit],
                                             self.isSlowHA_checkBox.isChecked())
                self.update_slow_volume_plot(False, fullUpdate=True)
            self.ref_slowCandlesUpdates.setData(self.data_slowCandles[self.index_slowUpdate_Limit-1:],
                                                self.isSlowHA_checkBox.isChecked())

            self.ref_slowCandles_rescale.setData(self.data_slowCandles[:, 0], self.data_slowCandles[:, 2],
                                                 pen=TRANS)

            _atr10 = ta.atr(self.data_slowCandles[:, (1, 4, 3, 2)], 10)
            _ema10 = ta.ema(self.data_slowCandles[:, 2], 10)
            _ema50 = ta.ema(self.data_slowCandles[:, 2], 50)
            _ema100 = ta.ema(self.data_slowCandles[:, 2], 100)
            _atr = ta.atr(self.data_slowCandles[:, (1, 4, 3, 2)], 5)
            _ema = ta.ema(self.data_slowCandles[:, 2], 5)
            _atr5per = 100*_atr[-1]/_ema[-1]

            self.ref_slowCurves[0].setData(self.data_slowCandles[:, 0],
                                               _ema10 - 2 * _atr10,
                                               pen=pg.mkPen(YELLOW, width=2))
            self.ref_slowCurves[1].setData(self.data_slowCandles[:, 0],
                                               _ema10 + 2 * _atr10,
                                               pen=pg.mkPen(BLUE, width=2))
            self.ref_slowCurves[2].setData(self.data_slowCandles[:, 0],
                                               _ema100,
                                               pen=pg.mkPen(DGREEN, width=2))

            self.ref_slowCandlesHlineLast.setPos(self.data_slowCandles[-1, 2])

            if autoscale:
                vb.autoRange()
                vb.enableAutoRange(axis='y', enable=True)

            self.timeUpdate_SlowCandle = time.perf_counter()

            end = time.perf_counter()
            try:
                self.fps_slowCandles_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_slowCandles_doubleSpinBox.setValue(999)

    def init_slow_volume_plot(self, figure):
        """initialize Slow Analysis plot"""
        # set style  
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("left").setTextPen(DGREY)
        figure.showGrid(x=True, y=True, alpha=0.2)

        p2 = pg.ViewBox()
        p_rescale = pg.ViewBox()
        vb = figure.getViewBox()
        figure.scene().addItem(p_rescale)
        p_rescale.setXLink(figure)
        p_rescale.setYLink(figure)

        p1 = figure.plotItem
        ay2 = pg.AxisItem("left")
        p1.layout.addItem(ay2, 2, 0)
        figure.scene().addItem(p2)
        ay2.linkToView(p2)
        p2.setXLink(figure)
        ay2.setZValue(1000)

        def update_views():
            p2.setGeometry(vb.sceneBoundingRect())
            p2.linkedViewChanged(vb, p2.XAxis)
            p_rescale.setGeometry(vb.sceneBoundingRect())
            p_rescale.linkedViewChanged(vb, p_rescale.XYAxes)
        update_views()
        vb.sigResized.connect(update_views)

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        self.ref_slowVolumeVline = vLine

        # +update/scale
        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                vLine.setPos(mousePoint.x())
                self.ref_slowCandlesVline.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())
        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        # plot
        width = 0.9*abs(self.data_slowBars[0, 0]-self.data_slowBars[1, 0])
        _zip = zip(self.data_slowCandles[1:, 2], self.data_slowCandles[:-1, 2])
        _brush_color = list(GREEN if el1 > el2 else (RED if el1 < el2 else YELLOW) for (el1, el2) in _zip)
        _brush_color.insert(0, YELLOW)

        volumeBars = pg.BarGraphItem(x=self.data_slowBars[:self.index_slowUpdate_Limit, 0],
                                     height=self.data_slowBars[:self.index_slowUpdate_Limit, 1],
                                     width=width,
                                     brushes=_brush_color[:self.index_slowUpdate_Limit], pen=TRANS)
        volumeBarsUpdates = pg.BarGraphItem(x=self.data_slowBars[self.index_slowUpdate_Limit:, 0],
                                     height=self.data_slowBars[self.index_slowUpdate_Limit:, 1],
                                     width=width,
                                     brushes=_brush_color[self.index_slowUpdate_Limit:], pen=TRANS)

        self.ref_slowBars_rescale = pg.PlotDataItem(self.data_slowBars[:, 0], self.data_slowBars[:, 1], pen=RED)

        figure.addItem(self.ref_slowBars_rescale)
        p_rescale.addItem(volumeBars)
        p_rescale.addItem(volumeBarsUpdates)

        self.ref_slowBars_viewBox = p_rescale
        self.ref_slowBars = volumeBars
        self.ref_slowBarsUpdates = volumeBarsUpdates

        _rate = np.divide(self.data_slowBars[:, 1], self.data_slowCurve[:, 1],
                          out=np.zeros_like(self.data_slowBars[:, 1], dtype=float),
                          where=self.data_slowCurve[:, 1] != 0)
        self.ref_slowCurve = pg.PlotDataItem(self.data_slowCurve[:, 0], _rate, pen=YELLOW)

        p2.addItem(self.ref_slowCurve)
        self.ref_slowCurve = [p2, self.ref_slowCurve]

        vb.setMouseEnabled(y=False)
        vb.setAutoVisible(y=True)
        vb.enableAutoRange(axis='y', enable=True)
        vb.setLimits(yMin=0)
        p2.setMouseEnabled(y=False)
        p2.setAutoVisible(y=True)
        p2.enableAutoRange(axis='y', enable=True)
        p2.invertY(True)
        p2.setLimits(yMin=0)

        figure.setAxisItems({'bottom': DateAxisItem.DateAxisItem()})

    def update_slow_volume_plot(self, autoscale=False, fullUpdate=False):
        """updates Slow Analysis plot: clear/plot new data"""
        start = time.perf_counter()

        if self.slowVolume_OnOff_Button.isChecked() and (abs(start-self.timeUpdate_SlowVolume) > 1/FRAME_RATE_GUI or fullUpdate):
            try:
                width = 0.9 * abs(self.data_slowBars[0, 0] - self.data_slowBars[1, 0])
            except:
                print("slow bars exception: ", self.data_slowBars)
            _zip = zip(self.data_slowCandles[1:, 2], self.data_slowCandles[:-1, 2])
            _brush_color = list(GREEN if el1 > el2 else (RED if el1 < el2 else YELLOW) for (el1, el2) in _zip)
            _brush_color.insert(0, YELLOW)

            if fullUpdate or self.index_slowUpdate_Limit < self.data_slowBars.shape[0] - 2 * CANDLES_UPDATE_LIMIT:
                self.ref_slowBars.setOpts(x=self.data_slowBars[:self.index_slowUpdate_Limit, 0],
                                             height=0.1*self.data_slowBars[:self.index_slowUpdate_Limit, 1],
                                             width=width,
                                             brushes=_brush_color[:self.index_slowUpdate_Limit], pen=TRANS)
            self.ref_slowBarsUpdates.setOpts(x=self.data_slowBars[self.index_slowUpdate_Limit:, 0],
                                                height=0.1*self.data_slowBars[self.index_slowUpdate_Limit:, 1],
                                                width=width,
                                                brushes=_brush_color[self.index_slowUpdate_Limit:], pen=TRANS)

            self.ref_slowBars_rescale.setData(self.data_slowBars[:, 0], 0.1*self.data_slowBars[:, 1], pen=TRANS)

            _rate = np.divide(self.data_slowBars[:, 1], self.data_slowCurve[:, 1],
                              out=np.zeros_like(self.data_slowBars[:, 1], dtype=float),
                              where=self.data_slowCurve[:, 1] != 0)
            self.ref_slowCurve[1].setData(self.data_slowCurve[:, 0], _rate, pen=YELLOW)

            # possible fix day/dates gaps:
            # ax = self.slowVolume_Plot.getAxis('bottom')
            # a = np.arange(self.data_slowBars[:, 0].size)
            # ticks = self.data_slowBars[:, 0]
            #
            # ax.plot(a, y)  # we plot y as a function of a, which parametrizes x
            # ax.setTicks([[(v, str(labels)) for v in ticks], []])  # set the ticks to be a

            if autoscale:
                vb = self.slowVolume_Plot.getViewBox()
                vb.autoRange()
                vb.enableAutoRange(axis='y', enable=True)
                self.ref_slowCurve[0].autoRange()
                self.ref_slowCurve[0].enableAutoRange(axis='y', enable=True)

            self.timeUpdate_SlowVolume = time.perf_counter()

            end = time.perf_counter()
            try:
                self.fps_slowVolume_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_slowVolume_doubleSpinBox.setValue(999)

    def init_fast_price_plot(self, figure):
        """initialize Fast plot"""
        #figure.setTitle("F-Price")
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("bottom").setStyle(showValues=False)
        figure.getAxis("left").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)
        vb = figure.getViewBox()

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        self.ref_fastCandlesVline = vLine
        self.ref_fastCandlesHline = hLine

        # +update/scale
        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                _dt = datetime.datetime.fromtimestamp(int(mousePoint.x())).strftime('%H:%M:%S')
                self.fastValues_lineEdit.setText(f"{_dt}/{mousePoint.y():.2f}")
                vLine.setPos(mousePoint.x())
                self.ref_fastVolumeVline.setPos(mousePoint.x())
                self.ref_orderBookHline.setPos(mousePoint.y())
                self.ref_VolumeProfHline.setPos(mousePoint.y())
                self.ref_level2ImageHline.setPos(mousePoint.y())
                hLine.setPos(mousePoint.y())
                self.data_orderBookBars_y = mousePoint.y()
        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        _atr = ta.atr(self.data_fastCandles[:, (1, 4, 3, 2)], 10)
        _ema = ta.ema(self.data_fastCandles[:, 2], 10)
        self.data_fastCurveATR[:, 1] = _ema - 2 * _atr
        self.data_fastCurveATR[:, 2] = _ema + 2 * _atr

        atr_down = pg.PlotDataItem(self.data_fastCurveATR[:, 0],
                                   self.data_fastCurveATR[:, 1],
                                   pen=pg.mkPen(YELLOW, width=2))
        atr_up = pg.PlotDataItem(self.data_fastCurveATR[:, 0],
                                 self.data_fastCurveATR[:, 2],
                                 pen=pg.mkPen(BLUE, width=2))
        ema = pg.PlotDataItem(self.data_fastCurveEMA[:, 0],
                                 self.data_fastCurveEMA[:, 1],
                                 pen=pg.mkPen(DGREEN, width=2))
        self.ref_fastCurves = [atr_down, atr_up, ema]
        figure.addItem(self.ref_fastCurves[0])
        figure.addItem(self.ref_fastCurves[1])
        figure.addItem(self.ref_fastCurves[2])

        bid = pg.PlotDataItem(self.data_fastCurveBid[:, 0], self.data_fastCurveBid[:, 1], pen=BLUE)
        ask = pg.PlotDataItem(self.data_fastCurveAsk[:, 0], self.data_fastCurveAsk[:, 1], pen=RED)
        self.ref_fastCurvesBA = [bid, ask]
        figure.addItem(self.ref_fastCurvesBA[0])
        figure.addItem(self.ref_fastCurvesBA[1])

        self.index_fastUpdate_Limit = max(self.data_fastCandles.shape[0] - CANDLES_UPDATE_LIMIT, 3)
        self.ref_fastCandles = CandlestickItem(self.data_fastCandles[:self.index_fastUpdate_Limit], self.isFastHA_checkBox.isChecked())
        self.ref_fastCandlesUpdates = CandlestickItem(self.data_fastCandles[self.index_fastUpdate_Limit-1:, :], self.isFastHA_checkBox.isChecked())
        figure.addItem(self.ref_fastCandles)
        figure.addItem(self.ref_fastCandlesUpdates)

        figure.setXLink(self.fastVolume_Plot)
        figure.setYLink(self.volumeProfile_Plot)

    def update_fast_price_plot(self, autoscale=False, fullUpdate=False):
        """updates Fast plot: clear/plot new data"""
        start = time.perf_counter()

        if self.fastPrice_OnOff_Button.isChecked() and (abs(start-self.timeUpdate_FastCandle) > 1/FRAME_RATE_GUI or fullUpdate):

            vb = self.fastPrice_Plot.getViewBox()
            [[xmin, _], [_, _]] = vb.viewRange()

            if self.historyValidate("TRADES_FAST") and self.historyValidate("BID") and self.historyValidate("ASK"):
                _lastT = self.data_fastCandles[-1, 0]
                vb.setXRange(min=xmin, max=1.05*_lastT-0.05*xmin, padding=0)

            if self.historyValidate("BID"):
                self.ref_fastCurvesBA[0].setData(self.data_fastCurveBid[:, 0], self.data_fastCurveBid[:, 1], pen=BLUE)
            if self.historyValidate("ASK"):
                self.ref_fastCurvesBA[1].setData(self.data_fastCurveAsk[:, 0], self.data_fastCurveAsk[:, 1], pen=RED)

            if self.historyValidate("TRADES_FAST"):
                if fullUpdate or self.index_fastUpdate_Limit < self.data_fastCandles.shape[0] - 2 * CANDLES_UPDATE_LIMIT:
                    # change index limit
                    self.index_fastUpdate_Limit = max(self.data_fastCandles.shape[0] - CANDLES_UPDATE_LIMIT, 3)
                    self.ref_fastCandles.setData(self.data_fastCandles[:self.index_fastUpdate_Limit],
                                                 self.isFastHA_checkBox.isChecked())
                    self.update_fast_volume_plot(False, fullUpdate=True)

                self.ref_fastCandlesUpdates.setData(self.data_fastCandles[self.index_fastUpdate_Limit-1:],
                                                    self.isFastHA_checkBox.isChecked())

                _atr10 = ta.atr(self.data_fastCandles[:, (1, 4, 3, 2)], 10)
                _ema10 = ta.ema(self.data_fastCandles[:, 2], 10)
                _ema50 = ta.ema(self.data_fastCandles[:, 2], 50)
                _atr = ta.atr(self.data_fastCandles[:, (1, 4, 3, 2)], 5)
                _ema = ta.ema(self.data_fastCandles[:, 2], 5)

                self.ref_fastCurves[0].setData(self.data_fastCandles[:, 0],
                                                   _ema10 - 2 * _atr10,
                                                   pen=pg.mkPen(YELLOW, width=2))
                self.ref_fastCurves[1].setData(self.data_fastCandles[:, 0],
                                                   _ema10 + 2 * _atr10,
                                                   pen=pg.mkPen(BLUE, width=2))
                self.ref_fastCurves[2].setData(self.data_fastCandles[:, 0],
                                                   _ema50,
                                                   pen=pg.mkPen(DGREEN, width=2))

            if autoscale:
                vb.autoRange()
                vb.enableAutoRange(axis='y', enable=True)

            self.timeUpdate_FastCandle = time.perf_counter()

            end = time.perf_counter()
            try:
                self.fps_fastCandles_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_fastCandles_doubleSpinBox.setValue(999)

    def init_fast_volume_plot(self, figure):
        """initialize Fast Volume plot"""
        # set style
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("left").setTextPen(DGREY)
        figure.showGrid(x=True, y=True, alpha=0.2)

        p2 = pg.ViewBox()
        p_rescale = pg.ViewBox()
        vb = figure.getViewBox()
        figure.scene().addItem(p_rescale)
        p_rescale.setXLink(figure)
        p_rescale.setYLink(figure)

        p1 = figure.plotItem
        ay2 = pg.AxisItem("left")
        p1.layout.addItem(ay2, 2, 0)
        figure.scene().addItem(p2)
        ay2.linkToView(p2)
        p2.setXLink(figure)
        ay2.setZValue(1000)

        def update_views():
            p2.setGeometry(vb.sceneBoundingRect())
            p2.linkedViewChanged(vb, p2.XAxis)
            p_rescale.setGeometry(vb.sceneBoundingRect())
            p_rescale.linkedViewChanged(vb, p_rescale.XYAxes)
        update_views()
        vb.sigResized.connect(update_views)

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        self.ref_fastVolumeVline = vLine

        # +update/scale
        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                vLine.setPos(mousePoint.x())
                self.ref_fastCandlesVline.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())
        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        # plot
        width = 0.9*abs(self.data_fastBars[0, 0] - self.data_fastBars[1, 0])
        _zip = zip(self.data_fastCandles[1:, 2], self.data_fastCandles[:-1, 2])
        _brush_color = list(GREEN if el1 > el2 else (RED if el1 < el2 else YELLOW) for (el1, el2) in _zip)
        _brush_color.insert(0, YELLOW)

        volumeBars = pg.BarGraphItem(x=self.data_fastBars[:self.index_fastUpdate_Limit, 0],
                                     height=self.data_fastBars[:self.index_fastUpdate_Limit, 1],
                                     width=width,
                                     brushes=_brush_color[:self.index_fastUpdate_Limit], pen=TRANS)
        volumeBarsUpdates = pg.BarGraphItem(x=self.data_fastBars[self.index_fastUpdate_Limit:, 0],
                                     height=self.data_fastBars[self.index_fastUpdate_Limit:, 1],
                                     width=width,
                                     brushes=_brush_color[self.index_fastUpdate_Limit:], pen=TRANS)

        self.ref_fastBars_rescale = pg.PlotDataItem(self.data_fastBars[:, 0], self.data_fastBars[:, 1], pen=RED)

        figure.addItem(self.ref_fastBars_rescale)
        p_rescale.addItem(volumeBars)
        p_rescale.addItem(volumeBarsUpdates)

        self.ref_fastBars_viewBox = p_rescale
        self.ref_fastBars = volumeBars
        self.ref_fastBarsUpdates = volumeBarsUpdates

        _rate = np.divide(self.data_fastBars[:, 1], self.data_fastCurve[:, 1],
                          out=np.zeros_like(self.data_fastBars[:, 1], dtype=float),
                          where=self.data_fastCurve[:, 1] != 0)
        self.ref_fastCurve = pg.PlotDataItem(self.data_fastCurve[:, 0], _rate, pen=YELLOW)
        p2.addItem(self.ref_fastCurve)
        self.ref_fastCurve = [p2, self.ref_fastCurve]

        vb.setMouseEnabled(y=False)
        vb.setAutoVisible(y=True)
        vb.enableAutoRange(axis='y', enable=True)
        vb.setLimits(yMin=0)
        p2.setMouseEnabled(y=False)
        p2.setAutoVisible(y=True)
        p2.enableAutoRange(axis='y', enable=True)
        p2.invertY(True)
        p2.setLimits(yMin=0)

        # p1.setAxisItems({'bottom': pg.DateAxisItem()})
        figure.setAxisItems({'bottom': DateAxisItem.DateAxisItem()})

    def update_fast_volume_plot(self, autoscale=False, fullUpdate=False):
        """updates Fast Volume plot: clear/plot new data"""
        start = time.perf_counter()

        if self.fastVolume_OnOff_Button.isChecked() and abs(start-self.timeUpdate_FastVolume) > 1/FRAME_RATE_GUI:
            try:
                width = 0.9*abs(self.data_fastBars[0, 0]-self.data_fastBars[1, 0])
            except:
                print("fast bars exception: ", self.data_fastBars)
            _zip = zip(self.data_fastCandles[1:, 2], self.data_fastCandles[:-1, 2])
            _brush_color = list(GREEN if el1 > el2 else (RED if el1 < el2 else YELLOW) for (el1, el2) in _zip)
            _brush_color.insert(0, YELLOW)

            if fullUpdate or self.index_fastUpdate_Limit < self.data_fastBars.shape[0] - 2 * CANDLES_UPDATE_LIMIT:
                self.ref_fastBars.setOpts(x=self.data_fastBars[:self.index_fastUpdate_Limit, 0],
                                             height=0.1*self.data_fastBars[:self.index_fastUpdate_Limit, 1],
                                             width=width,
                                             brushes=_brush_color[:self.index_fastUpdate_Limit], pen=TRANS)
            self.ref_fastBarsUpdates.setOpts(x=self.data_fastBars[self.index_fastUpdate_Limit:, 0],
                                                height=0.1*self.data_fastBars[self.index_fastUpdate_Limit:, 1],
                                                width=width,
                                                brushes=_brush_color[self.index_fastUpdate_Limit:], pen=TRANS)

            self.ref_fastBars_rescale.setData(self.data_fastBars[:, 0], 0.1*self.data_fastBars[:, 1], pen=TRANS)

            _t1 = time.perf_counter()
            _rate = np.divide(self.data_fastBars[:, 1], self.data_fastCurve[:, 1],
                              out=np.zeros_like(self.data_fastBars[:, 1], dtype=float),
                              where=self.data_fastCurve[:, 1] != 0)
            _t2 = time.perf_counter()

            self.ref_fastCurve[1].setData(self.data_fastCurve[:, 0], _rate, pen=YELLOW)

            if autoscale:
                vb = self.fastVolume_Plot.getViewBox()
                vb.autoRange()
                vb.enableAutoRange(axis='y', enable=True)
                self.ref_fastCurve[0].autoRange()
                self.ref_fastCurve[0].enableAutoRange(axis='y', enable=True)

            self.timeUpdate_FastVolume = time.perf_counter()

            end = time.perf_counter()
            try:
                self.fps_fastVolume_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_fastVolume_doubleSpinBox.setValue(999)

    def init_volume_profile_plot(self, figure):
        """initialize Volume Profile plot"""
        #figure.setTitle("V-Profile")
        figure.showAxis("top")
        figure.getAxis("top").setStyle(showValues=False)
        figure.showAxis("right")
        figure.getAxis("left").setStyle(showValues=False)
        figure.getAxis("right").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)

        p_rescale = pg.ViewBox()
        vb = figure.getViewBox()
        figure.scene().addItem(p_rescale)
        p_rescale.setXLink(figure)
        p_rescale.setYLink(figure)

        def update_views():
            p_rescale.setGeometry(vb.sceneBoundingRect())
            p_rescale.linkedViewChanged(vb, p_rescale.XYAxes)
        update_views()
        vb.sigResized.connect(update_views)

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        self.ref_VolumeProfVline = vLine
        self.ref_VolumeProfHline = hLine

        # +update/scale
        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                self.volumeProfValues_lineEdit.setText(f"{mousePoint.x():.1f}/{mousePoint.y():.2f}")
                vLine.setPos(mousePoint.x())
                self.ref_fastCandlesHline.setPos(mousePoint.y())
                self.ref_orderBookHline.setPos(mousePoint.y())
                self.ref_level2ImageHline.setPos(mousePoint.y())
                hLine.setPos(mousePoint.y())
                self.data_orderBookBars_y = mousePoint.y()
        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        # ## make some data
        buyPrice = self.data_volumeProfBuy[:, 0]
        buyVolume = self.data_volumeProfBuy[:, 1]
        sellPrice = self.data_volumeProfSell[:, 0]
        sellVolume = self.data_volumeProfSell[:, 1]

        # plot
        buyBars = pg.BarGraphItem(x0=0, y=buyPrice, height=self.priceMinIncrement * 0.9, x1=buyVolume, brush=LGREEN, pen=pg.mkPen(GREEN, width=2))
        sellBars = pg.BarGraphItem(x0=0, y=sellPrice, height=self.priceMinIncrement * 0.9, x1=sellVolume, brush=PINK, pen=pg.mkPen(RED, width=2))
        buyCurve = pg.PlotDataItem(buyVolume, buyPrice, pen=RED)
        sellCurve = pg.PlotDataItem(sellVolume, sellPrice, pen=RED)

        figure.addItem(buyCurve)
        figure.addItem(sellCurve)
        p_rescale.addItem(buyBars)
        p_rescale.addItem(sellBars)
        self.ref_volumeProfBars = [buyBars, sellBars]
        self.ref_volumeProfBars_rescale = [buyCurve, sellCurve]
        self.ref_volumeProfBars_viewBox = p_rescale

        vb.setMouseEnabled(x=False)
        vb.setAutoVisible(x=True)
        vb.enableAutoRange(axis='x', enable=True)

        # link to others
        figure.setYLink(self.mapLevel2_Plot)

    def update_volume_profile_plot(self, autoscale=False):
        """updates Volume Profile plot: clear/plot new data"""
        start = time.perf_counter()
        isSum = self.sumVolProf_checkBox.isChecked()

        if self.profileVolume_OnOff_Button.isChecked():
            if (not isSum and self.data_volumeProfBuy.size > 2 and self.data_volumeProfSell.size > 2) \
                    or (isSum and self.data_volumeProfSum.size):  # just one price, otherwise some problems with BarItem and len()
                if isSum:
                    sumPrice = self.data_volumeProfSum[:, 0]
                    sumVolume = self.data_volumeProfSum[:, 1]
                    dummyVolume = 0 * self.data_volumeProfSum[:, 1]
                    # plot
                    self.ref_volumeProfBars[0].setOpts(x0=0, y=sumPrice, height=self.priceMinIncrement * 0.9,
                                                       x1=sumVolume, brush=YELLOW, pen=pg.mkPen(ORANGE, width=2))
                    self.ref_volumeProfBars[1].setOpts(x0=0, y=sumPrice, height=self.priceMinIncrement * 0.9,
                                                       x1=dummyVolume, brush=TRANS, pen=pg.mkPen(TRANS, width=2))
                    self.ref_volumeProfBars_rescale[0].setData(sumVolume, sumPrice, pen=TRANS)
                    self.ref_volumeProfBars_rescale[1].setData(dummyVolume, sumPrice, pen=TRANS)
                else:
                    buyPrice = self.data_volumeProfBuy[:, 0]
                    buyVolume = self.data_volumeProfBuy[:, 1]
                    sellPrice = self.data_volumeProfSell[:, 0]
                    sellVolume = self.data_volumeProfSell[:, 1]
                    # plot
                    self.ref_volumeProfBars[0].setOpts(x0=0, y=buyPrice, height=self.priceMinIncrement * 0.9, x1=buyVolume, brush=LGREEN, pen=pg.mkPen(GREEN, width=2))
                    self.ref_volumeProfBars[1].setOpts(x0=0, y=sellPrice, height=self.priceMinIncrement * 0.9, x1=sellVolume, brush=PINK, pen=pg.mkPen(RED, width=2))
                    self.ref_volumeProfBars_rescale[0].setData(buyVolume, buyPrice, pen=TRANS)
                    self.ref_volumeProfBars_rescale[1].setData(sellVolume, sellPrice, pen=TRANS)

                if autoscale:
                    vb = self.volumeProfile_Plot.getViewBox()
                    vb.autoRange()
                    vb.enableAutoRange(axis='x', enable=True)

            end = time.perf_counter()
            try:
                self.fps_volumeProfile_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_volumeProfile_doubleSpinBox.setValue(999)

            self.timeUpdate_VolumeProf = time.perf_counter()

    def init_map_level2_plot(self, figure):
        """initialize Level2 plot"""
        #figure.setTitle("Level-2")
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("bottom").setStyle(showValues=False)
        figure.getAxis("left").setStyle(showValues=False)
        #figure.getAxis("right").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)
        vb = figure.getViewBox()

        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                self.data_level2Image_xyz[0] = mousePoint.x()
                self.data_level2Image_xyz[1] = mousePoint.y()

                self.ref_fastCandlesHline.setPos(mousePoint.y())
                self.ref_VolumeProfHline.setPos(mousePoint.y())
                self.ref_level2ImageHline.setPos(mousePoint.y())
                self.ref_orderBookHline.setPos(mousePoint.y())
                self.data_orderBookBars_y = mousePoint.y()

                self.mapValues_lineEdit.setText("%0.1f/%0.2f" % (self.data_level2Image_xyz[0],
                                                          self.data_level2Image_xyz[1]))
                self.ref_level2ImageVline.setPos(self.data_level2Image_xyz[0])
                self.ref_level2ImageHline.setPos(self.data_level2Image_xyz[1])

        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        self.ref_level2Image = pg.ImageItem()

        self.ref_level2Image.setLookupTable(cl.fireice)
        figure.addItem(self.ref_level2Image)

        # cross-hair section
        self.ref_level2ImageVline = pg.InfiniteLine(pen="c", angle=90, movable=False)
        self.ref_level2ImageHline = pg.InfiniteLine(pen="c", angle=0, movable=False)
        figure.addItem(self.ref_level2ImageVline, ignoreBounds=True)
        figure.addItem(self.ref_level2ImageHline, ignoreBounds=True)

        bid = pg.PlotDataItem(self.data_level2BACurves[:, 0], self.data_level2BACurves[:, 1], pen=BLUE)  # e-bid
        ask = pg.PlotDataItem(self.data_level2BACurves[:, 0], self.data_level2BACurves[:, 2], pen=RED)  # e-ask

        self.ref_level2BACurves = [bid, ask]
        figure.addItem(bid)
        figure.addItem(ask)

        _cond = self.data_level2Scatter[1:, 1] > self.data_level2Scatter[:-1, 1]
        _color = list(QtGui.QColor(GREEN) if element else QtGui.QColor(RED) for element in _cond)
        _color.insert(0, QtGui.QColor(YELLOW))

        size_curve = pg.PlotDataItem(self.data_level2Scatter[:, 0], self.data_level2Scatter[:, 1], pen=YELLOW)  # e-bid
        size_bars = pg.BarGraphItem(x=self.data_level2Scatter[:, 0], width=10,
                                                 y=self.data_level2Scatter[:, 1], height=self.data_level2Scatter[:, 1]/10,
                                                 brushes=_color, pen=pg.mkPen(TRANS, width=1))
        self.ref_level2Scatter = [size_curve, size_bars]

        figure.addItem(self.ref_level2Scatter[0])
        figure.addItem(self.ref_level2Scatter[1])

        figure.setYLink(self.orderBook_Plot)
        figure.setXLink(self.analysisLevel2_Plot)


    def update_map_level2_plot(self, autoscale=False):
        """updates Level2 plot: clear/plot new data"""
        start = time.perf_counter()
        _t_now = time.time()
        _size_norm = TRADES_SIZE_NORM * pow(10, self.tradesSize_verticalSlider.value()/10)

        if self.Level2map_OnOff_Button.isChecked() and abs(start-self.timeUpdate_Level2curves) > 1/FRAME_RATE_GUI:

            _t1 = time.perf_counter()

            if self.data_level2Scatter.shape[0] > 1:

                vb = self.mapLevel2_Plot.getViewBox()

                [[_, _], [ymin, ymax]] = vb.viewRange()

                if self.rescaleType_Price_comboBox.currentText() == "Last":
                    vb.setYRange(min=self.last_Price - 0.5 * (ymax - ymin), max=self.last_Price + 0.5 * (ymax - ymin),
                                 padding=0)
                elif self.rescaleType_Price_comboBox.currentText() == "Adaptive":
                    _b = 0.75  # cutof boundary , e.g. 0.75 = 50% of range
                    if (self.last_Price > ymin + _b * (ymax - ymin)) or (
                            self.last_Price < ymin + (1 - _b) * (ymax - ymin)):
                        vb.setYRange(min=self.last_Price - 0.5 * (ymax - ymin),
                                     max=self.last_Price + 0.5 * (ymax - ymin),
                                     padding=0)

                _text = self.updateRate_MarketDepth_comboBox.currentText()
                _w = 0.9*(0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1. if _text == "1 sec" else 0.1)
                self.data_level2Scatter_max = max(self.data_level2Scatter[:, 2])
                self.data_level2Scatter[:, 2] *= _size_norm / self.data_level2Scatter_max

                #if TRADES_GUI_BARS_NUMBER < self.data_level2Scatter.shape[0]:
                if TRADES_GUI_BARS_NUMBER <= self.data_level2Scatter[:, 0].size:
                    _cond = self.data_level2Scatter[-TRADES_GUI_BARS_NUMBER + 1:, 1] > self.data_level2Scatter[
                                                                                       -TRADES_GUI_BARS_NUMBER:-1, 1]
                else:
                    _cond = self.data_level2Scatter[1:, 1] > self.data_level2Scatter[:-1, 1]
                _color = list(QtGui.QColor(GREEN) if element else QtGui.QColor(RED) for element in _cond)
                #print("time pre-set Trades", round(1e3 * (time.perf_counter() - _t4), 3))
                _color.insert(0, QtGui.QColor(YELLOW))
                self.ref_level2Scatter[1].setOpts(x=self.data_level2Scatter[-TRADES_GUI_BARS_NUMBER:, 0] - _t_now, width=_w,
                                                            y=self.data_level2Scatter[-TRADES_GUI_BARS_NUMBER:, 1],
                                                            height=self.data_level2Scatter[-TRADES_GUI_BARS_NUMBER:, 2] * self.data_level2Scatter[-TRADES_GUI_BARS_NUMBER:, 1],
                                                            brushes=_color, pen=pg.mkPen(TRANS, width=_w/20))

                self.ref_level2Scatter[1].setOpacity(TRADES_BARS_OPACITY)
                self.data_level2Scatter[:, 2] *= self.data_level2Scatter_max / _size_norm

                self.ref_level2Scatter[0].setData(self.data_level2Scatter[-TRADES_GUI_NUMBER:, 0] - _t_now,
                                                  self.data_level2Scatter[-TRADES_GUI_NUMBER:, 1],
                                                  pen=pg.mkPen(YELLOW, width=2))
                #print("time set Trades", round(1e3 * (time.perf_counter() - _t4), 3))

            if self.data_level2BACurves.shape[0] > 1:
                self.ref_level2BACurves[0].setData(self.data_level2BACurves[-BA_GUI_NUMBER:, 0]-_t_now,
                                                   self.data_level2BACurves[-BA_GUI_NUMBER:, 1],
                                                   pen=pg.mkPen(BLUE, width=2))  # e-bid
                self.ref_level2BACurves[1].setData(self.data_level2BACurves[-BA_GUI_NUMBER:, 0]-_t_now,
                                                   self.data_level2BACurves[-BA_GUI_NUMBER:, 2],
                                                   pen=pg.mkPen(RED, width=2))  # e-ask

            # if autoscale:
            #     vb = self.mapLevel2_Plot.getViewBox()
            #     vb.autoRange()

            end = time.perf_counter()
            #print("time level2 curves update", round(1e3 * (end - start), 3))

            try:
                self.fps_level2map_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_level2map_doubleSpinBox.setValue(999)

            self.timeUpdate_Level2curves = time.perf_counter()

    def update_map_level2image_plot(self, autoscale=False):
        """updates Level2 plot: clear/plot new data"""
        start = time.perf_counter()
        _t_now = time.time()

        if self.Level2map_OnOff_Button.isChecked():

            _t1 = time.perf_counter()
            _time_steps = self.data_level2Image_TimeScale.shape[0]

            if _time_steps > 1:

                vb = self.mapLevel2_Plot.getViewBox()

                [[_, _], [ymin, ymax]] = vb.viewRange()
                if self.rescaleType_Price_comboBox.currentText() == "Last":
                    vb.setYRange(min=self.last_Price - 0.5 * (ymax - ymin), max=self.last_Price + 0.5 * (ymax - ymin),
                                 padding=0)
                elif self.rescaleType_Price_comboBox.currentText() == "Adaptive":
                    _b = 0.75  # cutof boundary , e.g. 0.75 = 50% of range
                    if (self.last_Price > ymin + _b * (ymax - ymin)) or (
                            self.last_Price < ymin + (1 - _b) * (ymax - ymin)):
                        vb.setYRange(min=self.last_Price - 0.5 * (ymax - ymin),
                                     max=self.last_Price + 0.5 * (ymax - ymin),
                                     padding=0)

                _dec = self.priceMinIncrement
                _out_min_price = self.last_Price - _dec * (LEVEL2_IMAGE_SHAPE_VIEW[0] - 1) / 2
                _out_max_price = self.last_Price + _dec * (LEVEL2_IMAGE_SHAPE_VIEW[0] - 1) / 2
                _out_num_price = LEVEL2_IMAGE_SHAPE_VIEW[0]

                if LEVEL2_IMAGE_SHAPE_VIEW[0] < LEVEL2_IMAGE_SHAPE_TICKS[0]:  # clipping is enabled

                    _ind_price_start = np.where(np.isclose(self.data_level2Image_PriceScale, _out_min_price))[0]
                    _ind_price_end = np.where(np.isclose(self.data_level2Image_PriceScale, _out_max_price))[0]
                    if len(_ind_price_start) == 0:  # update price is not in the price scale = take min possible
                        _ind_price_start = int(0)
                    else:
                        _ind_price_start = _ind_price_start[0]
                    if len(_ind_price_end) == 0:  # update price is not in the price scale = take max possible
                        _ind_price_end = int(LEVEL2_IMAGE_SHAPE_TICKS[0])
                    else:
                        _ind_price_end = _ind_price_end[0]
                else:
                    _ind_price_start = 0
                    _ind_price_end = LEVEL2_IMAGE_SHAPE_TICKS[0]

                _out_price = self.data_level2Image_PriceScale[_ind_price_start:_ind_price_end]
                _price_span = _out_price[-1] - _out_price[0] + self.priceMinIncrement

                # time clip
                _isTimeClipped = _time_steps > LEVEL2_IMAGE_SHAPE_VIEW[1]
                _ind_time_start = _time_steps-LEVEL2_IMAGE_SHAPE_VIEW[1] if _isTimeClipped else 0
                _time_scale = self.data_level2Image_TimeScale[_ind_time_start:] - _t_now  #self.data_level2Image_TimeScale[-1]
                _time_span = _time_scale[-1]-_time_scale[0]

                self.ref_level2Image.setImage(self.data_level2Image[_ind_price_start:_ind_price_end,
                                              _ind_time_start:_time_steps])  # , autoDownsample=False)
                _t2 = time.perf_counter()
                #print("time setimage", round(1e3 * (_t2 - _t1), 3))
                _min_lot = np.amin(self.data_level2Image[:, _time_steps-1])
                _max_lot = np.amax(self.data_level2Image[:, _time_steps-1])
                _max_abs_lot = max(abs(_max_lot), abs(_min_lot))
                #print("time search min/max", round(1e3 * (time.perf_counter() - _t2), 3))
                _color_norm = pow(10, self.colorScale_horizontalSlider.value()/10)
                self.ref_level2Image.setLevels([-_max_abs_lot/_color_norm, _max_abs_lot/_color_norm])
                _t3 = time.perf_counter()
                #print("time setlevels", round(1e3 * (_t3 - _t2), 3))
                # assumes linear time scaling, which is not the case for tick-based image
                self.ref_level2Image.setRect(
                    QRectF(_time_scale[0], _out_price[0] - self.priceMinIncrement / 2, _time_span,
                           _price_span))

            end = time.perf_counter()
            #print("time level 2 Image update", round(1e3 * (end - start), 3))
            try:
                self.fps_level2map_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_level2map_doubleSpinBox.setValue(999)

            self.timeUpdate_Level2map = time.perf_counter()

    def init_analysis_level2_plot(self, figure):
        """initialize Level2 Analysis plot"""
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("top").setStyle(showValues=False)
        figure.getAxis("right").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)
        p2 = pg.ViewBox()
        vb = figure.getViewBox()

        def update_views():
            p2.setGeometry(vb.sceneBoundingRect())
            p2.linkedViewChanged(vb, p2.XAxis)
        update_views()
        vb.sigResized.connect(update_views)

        p1 = figure.plotItem
        ay2 = pg.AxisItem("right")
        p1.layout.addItem(ay2, 2, 2)
        figure.scene().addItem(p2)
        ay2.linkToView(p2)
        p2.setXLink(figure)
        ay2.setZValue(100)

        Cumul = pg.PlotCurveItem(self.data_level2cumulDelta[:, 0], self.data_level2cumulDelta[:, 1], pen=BLUE)
        Vbid = pg.PlotCurveItem(self.data_level2sizeBACurves[:, 0], self.data_level2sizeBACurves[:, 1], pen=BLUE)  # dVbid
        Vask = pg.PlotCurveItem(self.data_level2sizeBACurves[:, 0], self.data_level2sizeBACurves[:, 2], pen=RED)  # dVask
        Vbid_Vask = pg.PlotCurveItem(self.data_level2imbalanceBACurve[:, 0], self.data_level2imbalanceBACurve[:, 1], pen=YELLOW)  # d(Vbid-Vask)

        self.ref_level2AnCurves = [Cumul, Vask, p2, Vbid_Vask]
        figure.addItem(Cumul)
        # figure.addItem(Vbid)
        # figure.addItem(Vask)
        p2.addItem(Vbid_Vask)
        #p2.addItem(hLine)

        vb.setMouseEnabled(y=False)
        vb.setAutoVisible(y=True)
        vb.enableAutoRange(axis='y', enable=True)
        p2.setMouseEnabled(y=False)
        p2.setAutoVisible(y=True)
        p2.enableAutoRange(axis='y', enable=True)

    def update_analysis_level2_plot(self, autoscale=False):
        """updates Analysis Level2 plot: clear/plot new data"""
        start = time.perf_counter()
        _t_now = time.time()
        vb = self.analysisLevel2_Plot.getViewBox()

        if self.Level2volume_OnOff_Button.isChecked() and abs(start-self.timeUpdate_Level2analysis) > 1/FRAME_RATE_GUI:

            [[xmin, _], [_, _]] = vb.viewRange()
            vb.setXRange(min=xmin, max=1, padding=0)

            if self.data_level2cumulDelta.shape[0] > 1:  # more than one record
                self.ref_level2AnCurves[0].setData(self.data_level2cumulDelta[:, 0] - _t_now,
                                                   self.data_level2cumulDelta[:, 1], pen=BLUE)

            # if self.data_level2sizeBACurves.shape[0] > 1:  # at least two data points
            #     self.ref_level2AnCurves[0].setData(
            #         self.data_level2sizeBACurves[-BA_GUI_NUMBER:, 0] - _t_now,
            #         self.data_level2sizeBACurves[-BA_GUI_NUMBER:, 1], pen=BLUE)  # dVbid
            #     self.ref_level2AnCurves[1].setData(
            #         self.data_level2sizeBACurves[-BA_GUI_NUMBER:, 0] - _t_now,
            #         self.data_level2sizeBACurves[-BA_GUI_NUMBER:, 2], pen=RED)  # dVask

            if self.data_level2imbalanceBACurve.shape[0] > 1:  # at least two data points
                self.ref_level2AnCurves[3].setData(
                    self.data_level2imbalanceBACurve[-BA_GUI_NUMBER:, 0] - _t_now,
                    self.data_level2imbalanceBACurve[-BA_GUI_NUMBER:, 1], pen=YELLOW)  # d(Vbid-Vask)

            if autoscale:
                vb.autoRange()
                vb.enableAutoRange(axis='y', enable=True)
                self.ref_level2AnCurves[2].autoRange()
                self.ref_level2AnCurves[2].enableAutoRange(axis='y', enable=True)

            end = time.perf_counter()
            try:
                self.fps_level2analysis_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_level2analysis_doubleSpinBox.setValue(999)

            self.timeUpdate_Level2analysis = end

    def init_order_book_plot(self, figure):
        """initialize OrderBook plot"""
        #figure.setTitle("Order Book")
        figure.showAxis("top")
        figure.showAxis("right")
        figure.getAxis("left").setStyle(showValues=False)
        figure.getAxis("right").setStyle(showValues=False)
        figure.getAxis("top").setStyle(showValues=False)
        figure.showGrid(x=True, y=True, alpha=0.2)

        p2 = pg.ViewBox()
        p_rescale = pg.ViewBox()
        vb = figure.getViewBox()
        figure.scene().addItem(p_rescale)
        p_rescale.setXLink(figure)
        p_rescale.setYLink(figure)

        self.orderBook_Plot.setAxisItems({"left": PercentAxisItem(zero_point=self.last_Price, orientation="left")})

        def update_views():
            p2.setGeometry(vb.sceneBoundingRect())
            p2.linkedViewChanged(vb, p2.YAxis)
            p_rescale.setGeometry(vb.sceneBoundingRect())
            p_rescale.linkedViewChanged(vb, p_rescale.XYAxes)
        update_views()
        vb.sigResized.connect(update_views)

        # cross-hair section
        vLine = pg.InfiniteLine(pen="c", angle=90, movable=False)
        hLine = pg.InfiniteLine(pen="c", angle=0, movable=False)
        hLineLast = pg.InfiniteLine(pen=pg.mkPen(WHITE, width=2), angle=0, movable=False)
        hLineLast_P1per = pg.InfiniteLine(pen="g", angle=0, movable=False)
        hLineLast_M1per = pg.InfiniteLine(pen="r", angle=0, movable=False)
        hLineLast_P3per = pg.InfiniteLine(pen="g", angle=0, movable=False)
        hLineLast_M3per = pg.InfiniteLine(pen="r", angle=0, movable=False)
        hLineBid = pg.InfiniteLine(pen="w", angle=0, movable=False)
        hLineAsk = pg.InfiniteLine(pen="w", angle=0, movable=False)
        self.ref_level2PositionHline = pg.InfiniteLine(pen="m", angle=0, movable=False)
        figure.addItem(self.ref_level2PositionHline, ignoreBounds=True)
        figure.addItem(vLine, ignoreBounds=True)
        figure.addItem(hLine, ignoreBounds=True)
        figure.addItem(hLineLast, ignoreBounds=True)
        # figure.addItem(hLineLast_P1per, ignoreBounds=True)
        # figure.addItem(hLineLast_M1per, ignoreBounds=True)
        # figure.addItem(hLineLast_P3per, ignoreBounds=True)
        # figure.addItem(hLineLast_M3per, ignoreBounds=True)
        figure.addItem(hLineBid, ignoreBounds=True)
        figure.addItem(hLineAsk, ignoreBounds=True)
        self.ref_orderBookHline = hLine
        self.ref_orderBookHlineLast = hLineLast
        self.ref_orderBookHlineLast_P1per = hLineLast_P1per
        self.ref_orderBookHlineLast_M1per = hLineLast_M1per
        self.ref_orderBookHlineLast_P3per = hLineLast_P3per
        self.ref_orderBookHlineLast_M3per = hLineLast_M3per
        self.ref_orderBookHlineBidAsk = [hLineBid, hLineAsk]

        self.ref_level2PositionHline.setPos(0)

        def mouse_moved(evt):
            pos = evt[0]  # using signal proxy turns original arguments into a tuple
            if figure.sceneBoundingRect().contains(pos):  # checks if within the plot
                mousePoint = vb.mapSceneToView(pos)
                mousePoint2 = p2.mapSceneToView(pos)
                self.orderBookValues_lineEdit.setText("%0.1f/%0.1f/%0.2f" % (mousePoint.x(), mousePoint2.x(), mousePoint.y()))
                vLine.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())
                self.ref_slowCandlesHline.setPos(mousePoint.y())
                self.ref_fastCandlesHline.setPos(mousePoint.y())
                self.ref_VolumeProfHline.setPos(mousePoint.y())
                self.ref_level2ImageHline.setPos(mousePoint.y())
                self.data_orderBookBars_y = mousePoint.y()

        figure.proxy = pg.SignalProxy(figure.scene().sigMouseMoved, rateLimit=CROSSHAIR_RATE_GUI, slot=mouse_moved)

        p1 = figure.plotItem
        ax2 = pg.AxisItem("bottom")
        p1.layout.addItem(ax2, 4, 1)
        figure.scene().addItem(p2)
        ax2.linkToView(p2)
        p2.setYLink(figure)
        ax2.setZValue(10000)

        # plot
        bidVolume = self.data_orderBookBidBars[:, 1]
        askVolume = self.data_orderBookAskBars[:, 1]
        bidPrice = self.data_orderBookBidBars[:, 0]
        askPrice = self.data_orderBookAskBars[:, 0]
        cumul_bidVolume = np.cumsum(np.flip(bidVolume))
        cumul_askVolume = np.cumsum(askVolume)
        cumul_bidPrice = np.flip(bidPrice)
        cumul_askPrice = askPrice

        bidBars = pg.BarGraphItem(x0=0, y=bidPrice, height=self.priceMinIncrement*0.9, x1=bidVolume, brush=LBLUE, pen=pg.mkPen(BLUE, width=2))
        askBars = pg.BarGraphItem(x0=0, y=askPrice, height=self.priceMinIncrement*0.9, x1=askVolume, brush=PINK, pen=pg.mkPen(RED, width=2))
        bidCurve = pg.PlotDataItem(bidVolume, bidPrice, pen=TRANS)
        askCurve = pg.PlotDataItem(askVolume, askPrice, pen=TRANS)

        figure.addItem(bidCurve)
        figure.addItem(askCurve)
        p_rescale.addItem(bidBars)
        p_rescale.addItem(askBars)

        self.ref_orderBookBars = [bidBars, askBars]
        self.ref_orderBookBars_rescale = [bidCurve, askCurve]
        self.ref_orderBookBars_viewBox = p_rescale

        #figure.setXRange(0, max(np.amax(bidVolume), np.amax(askVolume)), padding=0)
        cumulBid = pg.PlotCurveItem(cumul_bidVolume, cumul_bidPrice, pen=pg.mkPen("b", width=2))
        cumulAsk = pg.PlotCurveItem(cumul_askVolume, cumul_askPrice, pen=pg.mkPen("r", width=2))
        p2.addItem(cumulBid)
        p2.addItem(cumulAsk)
        p2.setXRange(0, max(np.amax(cumul_bidVolume), np.amax(cumul_askVolume)), padding=0)
        self.ref_orderBookCurves = [p2, cumulBid, cumulAsk]

        vb.invertX(True)
        vb.setMouseEnabled(x=False)
        vb.setAutoVisible(x=True)
        vb.enableAutoRange(axis='x', enable=True)
        vb.setLimits(xMin=0)
        p_rescale.invertX(True)
        p2.setMouseEnabled(x=False)
        p2.setAutoVisible(x=True)
        p2.enableAutoRange(axis='x', enable=True)
        p2.setLimits(xMin=0)

        figure.setYLink(self.mapLevel2_Plot)

    def update_order_book_plot(self, autoscale=False):
        """updates OrderBook plot: clear/plot new data"""
        start = time.perf_counter()

        if self.orderBook_OnOff_Button.isChecked() and abs(start-self.timeUpdate_OrderBook) > 1/FRAME_RATE_GUI:

            vb = self.orderBook_Plot.getViewBox()
            [[_, _], [ymin, ymax]] = vb.viewRange()

            self.orderBook_Plot.setAxisItems({"left": PercentAxisItem(zero_point=self.last_Price, orientation="left")})

            vb.setYRange(min=ymin + 1e-9, max=ymax, padding=0)  # hack to force change of axis labels

            self.ref_orderBookHlineBidAsk[0].setPos(self.last_avrgBidPrice)
            self.ref_orderBookHlineBidAsk[1].setPos(self.last_avrgAskPrice)

            _is_log = self.logOrderBook_checkBox.isChecked()
            _height = self.priceMinIncrement * 0.9

            _rate = self.ticker_rateVolume_doubleSpinBox.value()
            _rate = 10*_rate/60 if _rate > 0 else 1  # per second

            bidVolume = -self.data_orderBookBidBars[:, 1]  # makes sizes positive
            if bidVolume.shape[0] != 0:
                bidPrice = self.data_orderBookBidBars[:, 0]

                cumul_bidVolume = np.cumsum(bidVolume)/_rate
                cumul_bidPrice = bidPrice

                self.ref_orderBookBars[0].setOpts(x0=0, y=bidPrice, height=_height,
                                          x1=(np.log(bidVolume+1) if _is_log else bidVolume), brush=LBLUE,
                                          pen=pg.mkPen(BLUE, width=2))
                self.ref_orderBookBars_rescale[0].setData((np.log(bidVolume + 1) if _is_log else bidVolume), bidPrice,
                                                          pen=TRANS)
                self.ref_orderBookCurves[1].setData(cumul_bidVolume, cumul_bidPrice, pen=pg.mkPen("b", width=2))

            askVolume = self.data_orderBookAskBars[:, 1]

            if askVolume.shape[0] != 0:
                askPrice = self.data_orderBookAskBars[:, 0]

                cumul_askVolume = np.cumsum(askVolume)/_rate
                cumul_askPrice = askPrice

                self.ref_orderBookBars[1].setOpts(x0=0, y=askPrice, height=_height,
                                          x1=(np.log(askVolume+1) if _is_log else askVolume), brush=PINK,
                                          pen=pg.mkPen(RED, width=3))
                self.ref_orderBookBars_rescale[1].setData((np.log(askVolume + 1) if _is_log else askVolume), askPrice,
                                                          pen=TRANS)
                self.ref_orderBookCurves[2].setData(cumul_askVolume, cumul_askPrice, pen=pg.mkPen("r", width=2))

            self.levelsUsed_OrderBook_progressBar.setValue(int(0.5*100*(bidVolume.shape[0]+askVolume.shape[0])/MARKET_DEPTH_LEVELS))
            if autoscale:
                vb.autoRange()
                vb.enableAutoRange(axis='x', enable=True)
                self.ref_orderBookCurves[0].autoRange()
                self.ref_orderBookCurves[0].enableAutoRange(axis='x', enable=True)

            self.timeUpdate_OrderBook = time.perf_counter()

            end = time.perf_counter()
            try:
                self.fps_orderBook_doubleSpinBox.setValue(1/(end-start))
            except ZeroDivisionError:
                self.fps_orderBook_doubleSpinBox.setValue(999)

    def recalculate_VolumeProfile(self, fullUpdate=False, isSum=False):
        """recalculates/updates plot Volume profile data from
        Level2 Last trades tick based raw data @ span"""
        _start_t = time.perf_counter()

        _in_ticks = self.data_level2Scatter_ticks  # [time, price, size]
        _multiple = _in_ticks.size > 3
        _in_Buy = self.data_volumeProfBuy
        _in_Sell = self.data_volumeProfSell
        _in_Sum = self.data_volumeProfSum

        _dec = self.priceDecimals
        _text = self.span_VolumeProfile_comboBox.currentText()
        # brutal monkey patch to force full update every 100s
        fullUpdate = fullUpdate or ((_start_t-self.timeUpdate_fullVolumeProf) > VOL_PROF_CALC_TIME and _text !="All")

        if _multiple and (abs(_start_t-self.timeUpdate_VolumeProf) > 1/FRAME_RATE_GUI or fullUpdate):
            def Buy_Sell_histo(ind: np.array, in_BuySell: np.array,  buy_sell: int):
                """Outputs histograms of the buy/sell bars based on input array and indices to use"""
                if ind[0].size > 0:
                    _in_bars = copy.deepcopy(in_BuySell[ind, :][0])  # to fix weird 3d array
                    if _in_bars.size > 2:  # if more then one record
                        _in_bars[:, 0] = np.around(_in_bars[:, 0], decimals=_dec)
                        _out = _in_bars
                        _out_price, _indx, _counts = np.unique(_in_bars[:, 0], return_index=True,
                                                               return_counts=True)
                        _out_size = np.zeros((_out_price.size,), dtype=int)
                        # fill no dupes
                        _out_size[_counts == 1] = _in_bars[_indx[_counts == 1], 1]
                        # fill dupes
                        for i, el in enumerate(_out_price[_counts > 1]):
                            _ind_or = np.where(
                                np.isclose(_in_bars[:, 0], el))  # indices of duplicates within original array
                            _ind_new = np.where(
                                np.isclose(_out_price, el))  # indices of duplicates within new array
                            _out_size[_ind_new] = np.sum(_in_bars[_ind_or, 1])

                        return np.concatenate((_out_price.reshape(-1, 1), buy_sell * _out_size.reshape(-1, 1)),
                                              axis=1)
                    else:  # if one record
                        _out = _in_bars
                        _out[0, 0] = round(_out[0, 0], _dec)
                        _out[0, 1] = buy_sell * _out[0, 1]
                        return _out
                else: # no records
                    return np.array([])

            if not fullUpdate and ((not isSum and _in_Buy.size > 4 and _in_Sell.size > 4)
                                   or (isSum and _in_Sum.size > 4)):  # reasonably large input volume_prof bars
                if _text == "1 min":
                    _start = _in_ticks[-1, 0] - 60
                elif _text == "5 mins":
                    _start = _in_ticks[-1, 0] - 300
                elif _text == "10 mins":
                    _start = _in_ticks[-1, 0] - 600
                else:  # all
                    _start = _in_ticks[0, 0]

                # ------------add trades section
                # time cut of update:
                # update is shorter/equal than requested duration
                _isUpdateShort = _in_ticks[-self.data_volumeProf_update_size, 0] >= _start
                if _isUpdateShort:  # most likely
                    # has at least one element to spare for up/down condition
                    _tf = 1 if self.data_volumeProf_update_size < _in_ticks[:, 0].size else 0  # temp factor
                    _in_bars_BuySell = _in_ticks[-(self.data_volumeProf_update_size + _tf):, 1:]
                else:  # do as if full update (anyway only few trades)
                    # time cut:
                    _ind_time_cut = np.where(_in_ticks[:, 0] >= _start)[0][0]  # starting index
                    # has at least one element to spare for up/down condition
                    _tf = 1 if _ind_time_cut > 0 else 0  # temp factor
                    # inclusive index that defines start of summing in ticks (+1 if _tf = 0, i.e. no elements to spare)
                    self.data_volumeProf_time_cut_start_index = 1 + _ind_time_cut - _tf
                    _in_bars_BuySell = _in_ticks[(_ind_time_cut-_tf):, 1:]

                if isSum:
                    _ind_Sum = np.where(_in_bars_BuySell[1:, 1] >= -_in_bars_BuySell[:-1, 1])  # pointless, used for symmetry
                    _add_Sum = Buy_Sell_histo(_ind_Sum, _in_bars_BuySell, 1)
                else:
                    _ind_Buy = np.where(_in_bars_BuySell[1:, 1] >= _in_bars_BuySell[:-1, 1])
                    _add_Buy = Buy_Sell_histo(_ind_Buy, _in_bars_BuySell, 1)
                    _ind_Sell = np.where(_in_bars_BuySell[1:, 1] < _in_bars_BuySell[:-1, 1])
                    _add_Sell = Buy_Sell_histo(_ind_Sell, _in_bars_BuySell, -1)

                if _isUpdateShort:
                    # -------------reduce trades section
                    _ind_reduce = np.where(_in_ticks[:, 0] < _start)[0]
                    if _ind_reduce.size > 1:  # have something to reduce and not zero index
                        # time cut:
                        _ind_time_cut = _ind_reduce[-1]  # ending inclusive index
                        _tf = 1  # has one element to spare
                        _in_bars_BuySell = _in_ticks[self.data_volumeProf_time_cut_start_index-_tf:_ind_time_cut+1, 1:]
                        # inclusive index that defines start of summing in ticks
                        self.data_volumeProf_time_cut_start_index = _ind_time_cut + 1  # next after reduce segment

                        if isSum:
                            _ind_Sum = np.where(_in_bars_BuySell[1:, 1] >= -_in_bars_BuySell[:-1, 1])  # pointless, used for symmetry
                            _reduce_Sum = Buy_Sell_histo(_ind_Sum, _in_bars_BuySell, 1)
                        else:
                            _ind_Buy = np.where(_in_bars_BuySell[1:, 1] >= _in_bars_BuySell[:-1, 1])
                            _reduce_Buy = Buy_Sell_histo(_ind_Buy, _in_bars_BuySell, -1)
                            _ind_Sell = np.where(_in_bars_BuySell[1:, 1] < _in_bars_BuySell[:-1, 1])
                            _reduce_Sell = Buy_Sell_histo(_ind_Sell, _in_bars_BuySell, 1)
                    else:  # nothing to reduce
                        _reduce_Buy = np.array([])
                        _reduce_Sell = np.array([])
                        _reduce_Sum = np.array([])
                    # -------------include "add"/"reduce" trades section
                    def add_reduce(in0, in1):  # assumes in0 is 2d array with more than 2 elements, i.e. min 2x2
                        _in0 = copy.deepcopy(in0)
                        if len(in1.shape) > 1:  # more than one element (or one but 2d) and for can be used (2d array)
                            for i, el in enumerate(in1[:, 1]):  # loop over sizes/prices to add/reduce
                                _indx = np.where(np.isclose(_in0[:, 0], in1[i, 0]))  # prices are close
                                if len(_indx[0]) != 0:  # existing price
                                    _in0[_indx, 1] += el  # add/reduce sizes
                                else:
                                    _in0 = np.vstack((_in0, in1[i, :]))
                        elif in1.size > 0:  # one element and 1d
                            _indx = np.where(np.isclose(_in0[:, 0], in1[0]))  # prices are close
                            if len(_indx[0]) != 0:  # existing price
                                _in0[_indx, 1] += in1[1]  # add/reduce sizes
                            else:
                                _in0 = np.vstack((_in0, in1))
                        return _in0

                    if isSum:
                        _temp_Sum = add_reduce(_in_Sum, _add_Sum)
                        _out_Sum = add_reduce(_temp_Sum, _reduce_Sum)
                        # remove zero sizes and their prices
                        _ind_delete = np.where(_out_Sum[:, 1] < 1)
                        _out_Sum = np.delete(_out_Sum, _ind_delete, axis=0)
                    else:
                        _temp_Buy = add_reduce(_in_Buy, _add_Buy)
                        _temp_Sell = add_reduce(_in_Sell, _add_Sell)
                        _out_Buy = add_reduce(_temp_Buy, _reduce_Buy)
                        _out_Sell = add_reduce(_temp_Sell, _reduce_Sell)
                        # remove zero sizes and their prices
                        _ind_delete = np.where(_out_Buy[:, 1] < 1)
                        _out_Buy = np.delete(_out_Buy, _ind_delete, axis=0)
                        _ind_delete = np.where(_out_Sell[:, 1] > -1)
                        _out_Sell = np.delete(_out_Sell, _ind_delete, axis=0)
                else:  # update is longer than requested duration => use "add" arrays as "out" arrays
                    if isSum:
                        _out_Sum = _add_Sum
                    else:
                        _out_Buy = _add_Buy
                        _out_Sell = _add_Sell
            else:  # full update
                if _text == "1 min":
                    _start = _in_ticks[-1, 0] - 60
                elif _text == "5 mins":
                    _start = _in_ticks[-1, 0] - 300
                elif _text == "10 mins":
                    _start = _in_ticks[-1, 0] - 600
                else:
                    _start = _in_ticks[0, 0]
                # time cut:
                _ind_time_cut = np.where(_in_ticks[:, 0] >= _start)[0][0]  # starting index
                # has at least one element to spare for up/down condition
                _tf = 1 if _ind_time_cut > 0 else 0  # temp factor
                # inclusive index that defines start of summing in ticks (+1 if _tf = 0, i.e. no elements to spare)
                self.data_volumeProf_time_cut_start_index = 1 + _ind_time_cut - _tf

                _in_bars_BuySell = _in_ticks[(_ind_time_cut-_tf):, 1:]
                if isSum:
                    _ind_Sum = np.where(_in_bars_BuySell[1:, 1] >= -_in_bars_BuySell[:-1, 1])  # pointless, used for symmetry
                    _out_Sum = Buy_Sell_histo(_ind_Sum, _in_bars_BuySell, 1)
                else:
                    _ind_Buy = np.where(_in_bars_BuySell[1:, 1] >= _in_bars_BuySell[:-1, 1])
                    _out_Buy = Buy_Sell_histo(_ind_Buy, _in_bars_BuySell, 1)
                    _ind_Sell = np.where(_in_bars_BuySell[1:, 1] < _in_bars_BuySell[:-1, 1])
                    _out_Sell = Buy_Sell_histo(_ind_Sell, _in_bars_BuySell, -1)

                self.timeUpdate_fullVolumeProf = _start_t

            self.data_volumeProf_update_size = 0
            if isSum:
                self.data_volumeProfSum = _out_Sum
            else:
                self.data_volumeProfBuy = _out_Buy
                self.data_volumeProfSell = _out_Sell

            # print("time process Volume Profile", round(1e3*(time.perf_counter() - _start_t), 3),
            #       "for in shape", self.data_level2Scatter_ticks.shape)
            self.update_volume_profile_plot()

    def recalculate_Level2_BA(self, fullUpdate=False):
        """recalculates/updates plot from ticks and makes average data if time selected"""
        _start_t = time.perf_counter()

        if BA_CALC_NUMBER > self.data_level2BACurves_ticks.shape[0]:
            _in_ticks = self.data_level2BACurves_ticks  # (0,1,2)  # [time, bid ,ask]
        else:
            _in_ticks = self.data_level2BACurves_ticks[-BA_CALC_NUMBER:, :]  # (0,1,2)  # [time, bid ,ask]

        _text = self.updateRate_MarketDepth_comboBox.currentText()
        _multiple = _in_ticks.size > 5

        if _multiple and (abs(_start_t-self.timeUpdate_recalcLevel2BA) > 1 / FRAME_RATE_GUI or fullUpdate):
            if _text == "Ticks":
                self.data_level2BACurves = np.transpose([_in_ticks[:, 0], _in_ticks[:, 1], _in_ticks[:, 2]])
            else:
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.
                _start = _in_ticks[0, 0]
                _end = _in_ticks[-1, 0]

                _times_out = np.linspace(_start, _end, abs(int(round(1+(_end-_start)/_delta))))

                _func_b = interp1d(_in_ticks[:, 0], _in_ticks[:, 1], kind="nearest", copy=False,
                                   fill_value="extrapolate", assume_sorted=True)
                _func_a = interp1d(_in_ticks[:, 0], _in_ticks[:, 2], kind="nearest", copy=False,
                                   fill_value="extrapolate", assume_sorted=True)
                _price_b = _func_b(_times_out)
                _price_a = _func_a(_times_out)
                self.data_level2BACurves = np.transpose([_times_out, _price_b, _price_a])

            # print("time process BA", round(1e3 * (time.perf_counter() - _start_t), 3),
            #       "for in shape", self.data_level2BACurves_ticks.shape,
            #       "out shape", self.data_level2BACurves.shape)
            self.timeUpdate_recalcLevel2BA = time.perf_counter()
            self.update_map_level2_plot()

    def recalculate_Level2_Trades(self, fullUpdate=False):
        """recalculates/updates plot from ticks and makes sum data for average price if time selected"""
        # recalculating output array in case more than one element
        _start_t = time.perf_counter()

        _in_ticks = self.data_level2Scatter_ticks  # [time, price, size]
        _in_sampled = self.data_level2Scatter

        _multiple = _in_ticks.size > 3  # more than one record
        _multiple_sampled = _in_sampled.size > 6  # more than two records

        if _multiple and (abs(_start_t-self.timeUpdate_recalcLevel2Trades) > 1 / FRAME_RATE_GUI or fullUpdate):
            _text = self.updateRate_MarketDepth_comboBox.currentText()
            if _text == "Ticks":
                _out = _in_ticks
                self.data_level2Scatter = _out
            elif not fullUpdate and _multiple_sampled:
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.

                _index_last = np.where(_in_ticks[:, 0] > _in_sampled[-2, 0])[0][0]  # starting index
                _start = _in_sampled[-2, 0]  # start (inclusive because will be skipped) time bin
                _end = _in_sampled[-1, 0]  # end (inclusive) time bin
                while _end < _in_ticks[-1, 0]:  # increase _end time until larger-equal than last tick_time
                    _end += _delta

                # number of time bins to recalculate (i.e. update+calculate)
                _out_time_steps = abs(int(round(1+(_end-_start)/_delta)))
                _out_times = np.linspace(_start, _end, _out_time_steps)

                _func = interp1d(_in_ticks[_index_last:, 0], _in_ticks[_index_last:, 1], kind="nearest", copy=False,
                                 fill_value="extrapolate", assume_sorted=True)
                _price = _func(_out_times)
                _size = np.zeros((_out_times.size,), dtype=int)

                for i, el in enumerate(_in_ticks[_index_last:, 2]):  # loop over trades-> find indices where to add size
                    _below = (_out_times >= _in_ticks[_index_last+i, 0])
                    _above = (_out_times < _in_ticks[_index_last+i, 0])
                    _size[1 + np.where(_below[1:] & _above[:-1])[0]] += int(el)

                _indx = _size >= 1
                self.data_level2Scatter = np.concatenate(
                    (self.data_level2Scatter[:-1, :],
                     np.transpose([_out_times[_indx], _price[_indx], _size[_indx]])[1:, :]), axis=0)  # overwrite last record of old data and skip first record of new data

            else:  # full update
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.

                _start = _in_ticks[0, 0]
                _end = _in_ticks[-1, 0]

                # actual delta will be different but quite close for _end - _start >> delta
                _times_out = np.linspace(_start, _end, abs(int(round(1+(_end-_start)/_delta))))
                _func = interp1d(_in_ticks[:, 0], _in_ticks[:, 1], kind="nearest", copy=False,
                                 fill_value="extrapolate", assume_sorted=True)
                _price = _func(_times_out)
                _size = np.zeros((_times_out.size,), dtype=int)

                for i, el in enumerate(_in_ticks[:, 2]):  # loop over trades-> find indices where to add size
                    _below = (_times_out >= _in_ticks[i, 0])
                    _above = (_times_out < _in_ticks[i, 0])
                    _size[1 + np.where(_below[1:] & _above[:-1])[0]] += int(el)

                _indx = _size >= 1  # only non-zero sizes
                self.data_level2Scatter = np.transpose([_times_out[_indx], _price[_indx], _size[_indx]])

            # print("time process trades", round(1e3*(time.perf_counter() - _start_t), 3),
            #       "for in shape", self.data_level2Scatter_ticks.shape,
            #       "out shape", self.data_level2Scatter.shape)
            self.timeUpdate_recalcLevel2Trades = time.perf_counter()
            self.update_map_level2_plot()

    def recalculate_Level2_sizeBA(self, fullUpdate=False):  # currently not used
        """recalculates/updates plot from ticks and makes average data if time selected"""
        _start_t = time.perf_counter()

        if BA_CALC_NUMBER > self.data_level2BACurves_ticks.shape[0]:
            _in_ticks = self.data_level2BACurves_ticks  # (0,3,4)  # [time, bidSize ,askSize]
        else:
            _in_ticks = self.data_level2BACurves_ticks[-BA_CALC_NUMBER:, :]  # (0,3,4)  # [time, bidSize ,askSize]

        _text = self.updateRate_MarketDepth_comboBox.currentText()
        _multiple = _in_ticks.size > 5

        if _multiple and (abs(_start_t-self.timeUpdate_recalcLevel2BAsize) > 1 / FRAME_RATE_GUI or fullUpdate):
            if _text == "Ticks":
                self.data_level2sizeBACurves = np.transpose([_in_ticks[:, 0], _in_ticks[:, 3], _in_ticks[:, 4]])
            else:
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.
                _start = _in_ticks[0, 0]
                _end = _in_ticks[-1, 0]

                _times_out = np.linspace(_start, _end, abs(int(round(1+(_end-_start)/_delta))))

                _func_b = interp1d(_in_ticks[:, 0], _in_ticks[:, 3], kind="nearest", copy=False, assume_sorted=True)
                _func_a = interp1d(_in_ticks[:, 0], _in_ticks[:, 4], kind="nearest", copy=False, assume_sorted=True)
                _size_b = _func_b(_times_out)
                _size_a = _func_a(_times_out)
                self.data_level2sizeBACurves = np.transpose([_times_out, _size_b, _size_a])

            # print("time process sizeBA", round(1e3*(time.perf_counter() - _start_t), 3),
            #       "for in shape", self.data_level2BACurves_ticks.shape,
            #       "out shape", self.data_level2BACurves.shape)
            self.timeUpdate_recalcLevel2BAsize = time.perf_counter()
            self.update_analysis_level2_plot()

    def recalculate_Level2_cumulDelta(self, fullUpdate=False):
        """recalculates/updates plot from ticks and makes average data if time selected"""
        _start_t = time.perf_counter()
        _multiple = self.data_level2Scatter_ticks.size > 6  # more than two record

        if _multiple and (abs(_start_t-self.timeUpdate_recalcLevel2cumulDelta) > 1 / FRAME_RATE_GUI or fullUpdate):

            if CUMUL_CALC_NUMBER < self.data_level2Scatter_ticks.shape[0]:
                _in_ticks = self.data_level2Scatter_ticks[-CUMUL_CALC_NUMBER:, :-1]  # [time, price]
            else:
                _in_ticks = self.data_level2Scatter_ticks[:, :-1]

            _norm = 60. / self.cumulSpan_verticalSlider.value()
            _start = _in_ticks[0, 0]
            _end = _in_ticks[-1, 0]
            _nsteps_downsample = abs(int(round(1 + (_end - _start) / 1)))
            _times_out_downsample = np.linspace(_start, _end, _nsteps_downsample if _nsteps_downsample > 1 else 2)

            _deltas_frac = _in_ticks[1:, 1]/_in_ticks[:-1, 1] - 1  # in fraction of price
            _deltas = 100*(np.abs(_deltas_frac if self.absCumul_checkBox.isChecked() else _deltas_frac))  # in percents
            _cumul = np.cumsum(_deltas)
            _func = interp1d(_in_ticks[1:, 0], _cumul, kind="nearest", copy=False,
                             fill_value="extrapolate", assume_sorted=True)
            _cumul_samp = _func(_times_out_downsample)

            _delta_nwidth = min(_times_out_downsample.size-1, self.cumulSpan_verticalSlider.value())
            _times_samp = _times_out_downsample[_delta_nwidth:]
            _deltas_samp = _norm*(_cumul_samp[_delta_nwidth:]-_cumul_samp[:-_delta_nwidth])

            self.data_level2cumulDelta = np.transpose([_times_samp, _deltas_samp])
            self.cumulDeltaper_doubleSpinBox.setValue(_deltas_samp[-1])

            # print("time process cumulDelta", round(1e3*(time.perf_counter() - _start_t), 3),
            #       "for in shape", self.data_level2Scatter_ticks.shape,
            #       "out shape", self.data_level2cumulDelta.shape)
            self.timeUpdate_recalcLevel2cumulDelta = time.perf_counter()
            self.update_analysis_level2_plot()

    def recalculate_Level2_imbalanceBA(self, fullUpdate=False):
        """recalculates/updates plot from ticks and makes average data if time selected"""
        _start_t = time.perf_counter()

        # [time, sumbidSize ,sumaskSize]  # bidsizes are negative
        _mode = self.imbalanceMode_MarketDepth_comboBox.currentText()
        if _mode == "SumSize":
            _in_ticks = self.data_level2imbalanceSumSizeBACurve_ticks
        elif _mode == "SumCash":
            _in_ticks = self.data_level2imbalanceSumCashBACurve_ticks
        elif _mode == "AveragePrice":
            _in_ticks = self.data_level2imbalanceAvrgPriceBACurve_ticks   # [time, avrgbid, avrgAsk]
        else:  # force
            _in_ticks = self.data_level2imbalanceForceBACurve_ticks   # [time, forceBID, forceASK]

        if BA_CALC_NUMBER < _in_ticks.shape[0]:
            _in_ticks = _in_ticks[-BA_CALC_NUMBER:, :]

        _text = self.updateRate_MarketDepth_comboBox.currentText()
        _multiple = _in_ticks.size > 3

        if _multiple and (abs(_start_t-self.timeUpdate_recalcLevel2BAimbalance) > 1 / FRAME_RATE_GUI or fullUpdate):
            if _text == "Ticks":
                # (-bid)-ask=positive = stronger bids
                self.data_level2imbalanceBACurve = np.transpose([_in_ticks[:, 0],
                                                                 100*(-_in_ticks[:, 1] - _in_ticks[:, 2])/(-_in_ticks[:, 1] + _in_ticks[:, 2])])  # in percents of full
            else:  # full update or one/none elements in pre-plot array
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.
                _start = _in_ticks[0, 0]
                _end = _in_ticks[-1, 0]

                _times_out = np.linspace(_start, _end, abs(int(round(1+(_end-_start)/_delta))))
                # (-bid)-ask=positive = stronger bids
                _func = interp1d(_in_ticks[:, 0], 100*(-_in_ticks[:, 1] - _in_ticks[:, 2])/(-_in_ticks[:, 1] + _in_ticks[:, 2])  # in percents of full
                                 , kind="nearest", copy=False, assume_sorted=True)
                _size = _func(_times_out)
                self.data_level2imbalanceBACurve = np.transpose([_times_out, _size])

            # print("time process imbalanceBA", round(1e3*(time.perf_counter() - _start_t), 3),
            #       "for in shape", _in_ticks.shape,
            #       "out shape", self.data_level2imbalanceBACurve.shape)
            self.timeUpdate_recalcLevel2BAimbalance = time.perf_counter()
            self.update_analysis_level2_plot()

    def update_Map_ticks(self):
        """add converted raw data update to tick based arrays"""
        _flag = 0  # shifted: 0="none"/1="PRICE"/2="TIME"/3="Price&Time"
        _start_t = time.perf_counter()

        _dec = self.priceMinIncrement
        try:
            _in_update_price = copy.deepcopy(self.data_level2Image_update[:, 0])  # the closest to the spread is the first element
            _in_update_size = copy.deepcopy(self.data_level2Image_update[:, 1])
            _isUpdateBid = True if _in_update_size[0] < 0 else False

            _in_update_mid_price = _in_update_price[0]

            if _isUpdateBid:  # to make price increase, that is: min/max [0]/[-1]
                _in_update_price = np.flip(_in_update_price)
                _in_update_size = np.flip(_in_update_size)
        except:
            print("Exception in Update Map, with update shape:", self.data_level2Image_update.shape)
            print(self.data_level2Image_update)

        _in_time = self.data_level2Image_TimeScale_ticks
        _in_time_steps = _in_time.size
        _in_current_index = _in_time_steps-1  # current time index (i.e. index of the update: time is filled but not data image yet)

        if _in_time_steps > 1:  # not the first run
            # price shift only check after 3 updates (to exclude false triggers at the start of updates)
            if _in_time_steps > 3:
                                # current mid price above 75% of previous price scale = rescale and shift data
                if _in_update_mid_price > self.data_level2Image_PriceScale[int((LEVEL2_IMAGE_SHAPE_TICKS[0]-1)*3/4)]:
                    _flag += 1
                    _out_size = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, dtype=LEVEL2_IMAGE_TYPE)
                    _indx = np.where(np.isclose(self.data_level2Image_PriceScale, _in_update_mid_price))

                    _out_min_price = _in_update_mid_price - _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
                    _out_max_price = _in_update_mid_price + _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
                    self.data_level2Image_PriceScale = np.linspace(_out_min_price, _out_max_price,
                                                                   LEVEL2_IMAGE_SHAPE_TICKS[0])
                    if len(_indx[0]) > 0:  # new center price within the price range
                        _in_update_mid_price_index = _indx[0][0]
                        # number of indices to shift
                        _price_shift = _in_update_mid_price_index - int((LEVEL2_IMAGE_SHAPE_TICKS[0] - 1) * 1 / 2)
                        _out_size[:-_price_shift, :_in_current_index] = copy.deepcopy(self.data_level2Image_ticks[_price_shift:,
                                                                                      :_in_current_index])
                        self.data_level2Image_ticks = _out_size
                        #print("Image Level2: price shifted up")
                        logging.info("Image Level2: price shifted up")
                    else:
                        print("Image Level2: price shifted up too fast: ERROR")
                        logging.error("Image Level2: price shifted up too fast")

                # current mid price below 25% of previous price scale = rescale and shift data
                elif _in_update_mid_price < self.data_level2Image_PriceScale[int((LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/4)]:
                    _flag += 1
                    _out_size = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, dtype=LEVEL2_IMAGE_TYPE)
                    _indx = np.where(np.isclose(self.data_level2Image_PriceScale, _in_update_mid_price))
                    _out_min_price = _in_update_mid_price - _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
                    _out_max_price = _in_update_mid_price + _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
                    self.data_level2Image_PriceScale = np.linspace(_out_min_price, _out_max_price,
                                                                   LEVEL2_IMAGE_SHAPE_TICKS[0])
                    if len(_indx[0]) > 0:  # new center price within the price range
                        _in_update_mid_price_index = _indx[0][0]
                        # number of indices to shift
                        _price_shift = int((LEVEL2_IMAGE_SHAPE_TICKS[0] - 1) / 2) - _in_update_mid_price_index
                        _out_size[_price_shift:, :_in_current_index] = copy.deepcopy(self.data_level2Image_ticks[:-_price_shift,
                                                                                     :_in_current_index])
                        self.data_level2Image_ticks = _out_size
                        #print("Image Level2: price shifted down")
                        logging.info("Image Level2: price shifted down")
                    else:
                        print("Image Level2: price shifted down too fast: ERROR")
                        logging.error("Image Level2: price shifted down too fast")

            _in_price_scale = self.data_level2Image_PriceScale
            _in_ticks = self.data_level2Image_ticks

            # time shift
            if _in_current_index >= int((LEVEL2_IMAGE_SHAPE_TICKS[1])*3/4):  # current time above 75% the time scale scale = rescale and shift data
                _flag += 2

                _out_size = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, dtype=LEVEL2_IMAGE_TYPE)
                _time_shift = int((LEVEL2_IMAGE_SHAPE_TICKS[1])/2)  # number of indices to shift/drop: 50% of time_Scale
                # copy 50% of data (including current index/update)
                _out_size[:, :_in_current_index - _time_shift] = copy.deepcopy(
                    self.data_level2Image_ticks[:, _time_shift:_in_current_index])
                self.data_level2Image_ticks = _out_size

                self.data_level2Image_TimeScale_ticks = copy.deepcopy(
                    self.data_level2Image_TimeScale_ticks[_time_shift:_in_current_index + 1])

                logging.info("Image Level2: time shifted")

                _in_time = self.data_level2Image_TimeScale_ticks
                _in_time_steps = _in_time.size
                _in_current_index = _in_time_steps - 1

            _s0_t = time.perf_counter()
            # update with previous data
            if _isUpdateBid:
                _ind_last_asks = np.where(_in_ticks[:, _in_current_index-1] >= 1)
                # previous column has asks (should be almost always except first several columns)
                if _ind_last_asks[0].shape[0] > 0:
                    _in_last_unique_price_asks = _in_price_scale[_ind_last_asks]
                    # delete those that below-equal highest bid price
                    _ind_delete = np.where(_in_last_unique_price_asks <= _in_update_price[-1])
                    _ind_last_asks = np.delete(_ind_last_asks, _ind_delete)  # only valid asks/indices

                    self.data_level2Image_ticks[_ind_last_asks, _in_current_index] = _in_ticks[
                        _ind_last_asks, _in_current_index - 1]
            elif not _isUpdateBid:
                _ind_last_bids = np.where(_in_ticks[:, _in_current_index-1] <= -1)
                # previous column has bids (should be almost always except first several columns)
                if _ind_last_bids[0].shape[0] > 0:
                    _in_last_unique_price_bids = _in_price_scale[_ind_last_bids]
                    # delete those that above-equal lowest bid price
                    _ind_delete = np.where(_in_last_unique_price_bids >= _in_update_price[0])
                    _ind_last_bids = np.delete(_ind_last_bids, _ind_delete)  # only valid bids/indices

                    self.data_level2Image_ticks[_ind_last_bids, _in_current_index] = _in_ticks[
                        _ind_last_bids, _in_current_index - 1]
            _s1_t = time.perf_counter()
            #print("time process: single entry old copy", round(1e3 * (_s1_t - _s0_t), 3))

            # fill with new data: seems to be the most time consuming
            for i, el in enumerate(_in_update_size):  # speed is limited by active price levels and possibly pricescale
                _indx = np.where(np.isclose(self.data_level2Image_PriceScale, _in_update_price[i]))
                # update price is in the price scale (because can be clipped by image size limitations and out of rangebid askBars
                if len(_indx[0]) > 0:
                    self.data_level2Image_ticks[_indx[0], _in_current_index] = el
            _s2_t = time.perf_counter()
            #print("time process: single entry new copy", round(1e3 * (_s2_t - _s1_t), 3))

        else:
            _s1_t = time.perf_counter()

            _out_min_price = _in_update_mid_price - _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
            _out_max_price = _in_update_mid_price + _dec*(LEVEL2_IMAGE_SHAPE_TICKS[0]-1)/2
            _out_num_cells = LEVEL2_IMAGE_SHAPE_TICKS[0]
            _out_price = np.linspace(_out_min_price, _out_max_price, _out_num_cells)
            _out_size = np.zeros((_out_num_cells, ), dtype=LEVEL2_IMAGE_TYPE)

            for i, el in enumerate(_in_update_size):  # fill with new data
                _indx = np.where(np.isclose(_out_price, _in_update_price[i]))
                _out_size[_indx] = el

            #print("out_ticks shape", _out_size.shape)
            self.data_level2Image_ticks[:, _in_current_index] = _out_size
            self.data_level2Image_PriceScale = _out_price

            _s2_t = time.perf_counter()
            #print("time process: first entry combine", round(1e3 * (_s2_t - _s1_t), 3))

        _out = self.data_level2Image_ticks[:, _in_current_index]
        #print(f"time process add to Map {round(1e3 * (time.perf_counter() - _start_t), 3)} for index {_in_current_index}")

        return False if _flag == 0 else True

    def recalculate_Map(self, fullUpdate=False):
        """recalculates/updates plot of OrderBook history from Market depth raw data @ time step"""
        _start_t = time.perf_counter()

        _dec = self.priceMinIncrement
        _text = self.updateRate_MarketDepth_comboBox.currentText()
        _isLog_map = self.logMap_checkBox.isChecked()

        _in_time = self.data_level2Image_TimeScale
        _in_time_steps = _in_time.size
        _in_current_index = _in_time_steps-1  # current time index (i.e. pre-update)

        _in_time_ticks = self.data_level2Image_TimeScale_ticks
        _in_time_ticks_steps = _in_time_ticks.size
        _in_current_ticks_index = _in_time_ticks_steps-1  # current time_ticks index

        _in_ticks = self.data_level2Image_ticks

        if _in_time_ticks_steps > 1 and (abs(_start_t-self.timeUpdate_Level2map) > 1/FRAME_RATE_GUI or fullUpdate):

            if _text == "Ticks" and not _isLog_map:
                self.data_level2Image = _in_ticks

                self.data_level2Image_TimeScale = _in_time_ticks

            elif _text == "Ticks" and _isLog_map:
                if fullUpdate:
                    self.data_level2Image = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, float)   # placeholder
                    for _col in range(_in_time_ticks_steps):
                        _mask_ask = np.where(_in_ticks[:, _col] >= 1)
                        _mask_bid = np.where(_in_ticks[:, _col] <= -1)

                        self.data_level2Image[_mask_ask, _col] = np.log(1+_in_ticks[_mask_ask, _col])
                        self.data_level2Image[_mask_bid, _col] = -np.log(1+np.abs(_in_ticks[_mask_bid, _col]))

                        # if _nPrices > 1:
                        #     self.data_level2Image[:, _col] = savgol_filter(self.data_level2Image[:, _col], _nPrices, 2)

                elif not fullUpdate:
                    for _col in range(self.data_orderBook_update_size):
                        _mask_ask = np.where(_in_ticks[:, _in_current_ticks_index-_col] >= 1)
                        _mask_bid = np.where(_in_ticks[:, _in_current_ticks_index-_col] <= -1)
                        self.data_level2Image[_mask_ask, _in_current_ticks_index - _col] = np.log(
                            1+_in_ticks[_mask_ask, _in_current_ticks_index - _col])
                        self.data_level2Image[_mask_bid, _in_current_ticks_index - _col] = -np.log(
                            1+np.abs(_in_ticks[_mask_bid, _in_current_ticks_index - _col]))

                        # if _nPrices > 1:
                        #     self.data_level2Image[:, _in_current_ticks_index - _col] = savgol_filter(
                        #         self.data_level2Image[:, _in_current_ticks_index - _col], _nPrices, 2)

                self.data_level2Image_TimeScale = _in_time_ticks

            elif _text != "Ticks" and not fullUpdate and _in_time_steps > 2:
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.

                # controls quality of interpolation: more = better, but longer (actual update only one)
                _TICKS_NUMBER = self.data_orderBook_update_size + 5

                _t1 = time.perf_counter()

                # defines index in time_ticks to include in recalculation
                _index_last = _in_current_ticks_index - _TICKS_NUMBER + 1
                # in case elements less than "_TICKS_NUMBER" constant
                _index_last = 0 if _index_last < 0 else _index_last

                _start = _in_time[-1]  # start (inclusive) time bin
                for i, el in reversed(list(enumerate(_in_time))):
                    if el < _in_time_ticks[_index_last]:
                        _start = el+_delta
                        break

                _end = _in_time[-1]  # end (inclusive) time bin
                while _end < _in_time_ticks[-1]:
                    _end += _delta

                # number of time bins to update (i.e. not including the new time bins)
                _bins_update = int(round(1+(_in_time[-1]-_start)/_delta))
                # number of time bins to recalculate (i.e. update+calculate)
                _out_time_steps = int(round(1 + (_end - _start) / _delta))
                _out_times = np.linspace(_start, _end, _out_time_steps)

                _t2 = time.perf_counter()
                #print("time process index search", round(1e3 * (_t2 - _t1), 3))

                if _isLog_map:
                    _log_image = np.zeros(_in_ticks[:, _index_last:_in_time_ticks_steps].shape, float)  # placeholder
                    for _col in range(_in_time_ticks[_index_last:].size):
                        _mask_ask = np.where(_in_ticks[:, _index_last + _col] >= 1)
                        _mask_bid = np.where(_in_ticks[:, _index_last + _col] <= -1)

                        _log_image[_mask_ask, _col] = np.log(1+_in_ticks[_mask_ask, _index_last + _col])
                        _log_image[_mask_bid, _col] = -np.log(1+np.abs(_in_ticks[_mask_bid, _index_last + _col]))

                    _func = interp1d(_in_time_ticks[_index_last:_in_time_ticks_steps], _log_image, kind="previous",
                                     axis=1, copy=False, fill_value="extrapolate", assume_sorted=True)
                    _out_size = _func(_out_times)
                else:
                    _func = interp1d(_in_time_ticks[_index_last:_in_time_ticks_steps],
                                     _in_ticks[:, _index_last:_in_time_ticks_steps], kind="previous",
                                     axis=1, copy=False, fill_value="extrapolate", assume_sorted=True)
                    _out_size = _func(_out_times).astype(LEVEL2_IMAGE_TYPE)

                _t4 = time.perf_counter()
                # print("time process stack", round(1e3 * (_t4 - _t3), 3))

                self.data_level2Image_TimeScale = np.concatenate((_in_time[:-_bins_update], _out_times))
                # change current index based on the update size
                _out_current_index = _in_current_index - _bins_update + _out_time_steps
                # add update=start with next to current index
                # fixes problem that may throw out of limits when single update is more than 25% of the time steps number (for 100ms and 1000/4000 points =100 seconds update)
                _broadcast_to = (_out_current_index + 1) - ((_in_current_index + 1) - _bins_update)
                _broadcast_from = _out_time_steps
                try:
                    self.data_level2Image[:, (_in_current_index + 1) - _bins_update:_out_current_index + 1] = _out_size[:, -min(_broadcast_to, _broadcast_from):]

                    # time shift/delete half of previous
                    if self.data_level2Image_TimeScale.size > int(LEVEL2_IMAGE_SHAPE[1]*3/4):
                        self.data_level2Image_TimeScale = np.delete(self.data_level2Image_TimeScale,
                                                                    np.s_[:int(LEVEL2_IMAGE_SHAPE[1] / 2)])
                        # keep some data (with updated current index)
                        _temp = copy.deepcopy(self.data_level2Image[:,
                                              _out_current_index + 1 - self.data_level2Image_TimeScale.size:_out_current_index + 1])
                        self.data_level2Image = np.zeros(LEVEL2_IMAGE_SHAPE, LEVEL2_IMAGE_TYPE)  # re-zero image
                        self.data_level2Image[:, :self.data_level2Image_TimeScale.size] = _temp  # put back some data

                        logging.info("Image Level2 sampled: time shifted")
                        # print("Image Level2 sampled: time shifted")
                except:
                    print("Error with broadcasting image")
                    logging.warning("Error with broadcasting image")
                _t5 = time.perf_counter()
                #print("time process full update", round(1e3 * (_t5 - _t1), 3))

            else: # _text != "Ticks" and (fullUpdate or _in_time_steps <= 2):
                _delta = 0.1 if _text == "100 msec" else 0.25 if _text == "250 msec" else 0.5 if _text == "500 msec" else 1.

                self.data_level2Image = np.zeros(LEVEL2_IMAGE_SHAPE, LEVEL2_IMAGE_TYPE)

                _start = _in_time_ticks[0]
                _end = _in_time_ticks[-1]
                _out_time_steps = int(round(1+(_end-_start)/_delta))
                _out_times = np.linspace(_start, _end, _out_time_steps)

                if _out_time_steps > int(LEVEL2_IMAGE_SHAPE[1]*3/4):  # time shift
                    _out_times = copy.deepcopy(_out_times[-int(LEVEL2_IMAGE_SHAPE[1]/4):])
                    _out_time_steps = _out_times.size

                if _isLog_map:
                    _log_image = np.zeros(LEVEL2_IMAGE_SHAPE_TICKS, float)  # placeholder
                    for _col in range(_in_time_ticks_steps):
                        _mask_ask = np.where(_in_ticks[:, _col] >= 1)
                        _mask_bid = np.where(_in_ticks[:, _col] <= -1)

                        _log_image[_mask_ask, _col] = np.log(1+_in_ticks[_mask_ask, _col])
                        _log_image[_mask_bid, _col] = -np.log(1+np.abs(_in_ticks[_mask_bid, _col]))

                    _func = interp1d(_in_time_ticks, _log_image[:, :_in_time_ticks_steps], kind="previous", copy=False,
                                     assume_sorted=True)
                    _out_size = _func(_out_times)
                else:
                    _func = interp1d(_in_time_ticks, _in_ticks[:, :_in_time_ticks_steps], kind="previous", copy=False,
                                     assume_sorted=True)
                    _out_size = _func(_out_times).astype(LEVEL2_IMAGE_TYPE)

                self.data_level2Image[:, :_out_time_steps] = _out_size
                self.data_level2Image_TimeScale = _out_times

            self.data_orderBook_update_size = 0
            _end_t = time.perf_counter()
            #print("time process Map", round(1e3*(_end_t - _start_t), 3))
            self.update_map_level2image_plot()

    def recalculate_sumBidAsk(self, _time):
        """updates summed Bid/Ask from Order book Bid/Ask tick data update from Market depth"""
        _start_t = time.perf_counter()

        _update_force = np.array([_time, self.last_forceBid, self.last_forceAsk])  # negative bids
        _update_sum_size = np.array([_time, self.last_sumBidSize, self.last_sumAskSize])  # negative bids
        _update_sum_price = np.array([_time, self.last_sumPriceBidSize, self.last_sumPriceAskSize])  # negative bids
        _update_avrg = np.array([_time, -100 * (self.last_avrgAskPrice / self.last_Price - 1),  # negative asks (=larger asks-bids-asks=better)
                                 100 * (1 - self.last_avrgBidPrice / self.last_Price)])  # positive bids

        if self.data_level2imbalanceSumSizeBACurve_ticks.shape[0] == 0:
            self.data_level2imbalanceSumSizeBACurve_ticks = _update_sum_size
        else:
            self.data_level2imbalanceSumSizeBACurve_ticks = np.vstack((self.data_level2imbalanceSumSizeBACurve_ticks, _update_sum_size))

        if self.data_level2imbalanceSumCashBACurve_ticks.shape[0] == 0:
            self.data_level2imbalanceSumCashBACurve_ticks = _update_sum_price
        else:
            self.data_level2imbalanceSumCashBACurve_ticks = np.vstack((self.data_level2imbalanceSumCashBACurve_ticks, _update_sum_price))

        if self.data_level2imbalanceAvrgPriceBACurve_ticks.shape[0] == 0:
            self.data_level2imbalanceAvrgPriceBACurve_ticks = _update_avrg
        else:
            self.data_level2imbalanceAvrgPriceBACurve_ticks = np.vstack(
                (self.data_level2imbalanceAvrgPriceBACurve_ticks, _update_avrg))

        if self.data_level2imbalanceForceBACurve_ticks.shape[0] == 0:
            self.data_level2imbalanceForceBACurve_ticks = _update_force
        else:
            self.data_level2imbalanceForceBACurve_ticks = np.vstack((self.data_level2imbalanceForceBACurve_ticks,
                                                                     _update_force))

        _end_t = time.perf_counter()
        #print("time process sumBidAsk", round(1e3 * (_end_t - _start_t), 3))
        self.recalculate_Level2_imbalanceBA()

    def remove_dupes_BidAsk(self, isBid=True):
        """removes duplicates due to price rounding"""
        _start_t = time.perf_counter()

        if isBid:
            _in_price = self.data_orderBookBidBars[:, 0]
            _in_size = -self.data_orderBookBidBars[:, 1]   # MAKES BID SIZES NEGATIVE
        else:
            _in_price = self.data_orderBookAskBars[:, 0]
            _in_size = self.data_orderBookAskBars[:, 1]

        _non_z = np.nonzero(_in_price)
        _in_price = _in_price[_non_z]
        _in_size = _in_size[_non_z]

        _price_sorted, _indx, _counts = np.unique(_in_price, return_index=True,
                                                        return_counts=True)
        _out_size = np.zeros((_price_sorted.size,), dtype=int)

        # fill no dupes
        _out_size[_counts == 1] = _in_size[_indx[_counts == 1]]
        # fill dupes
        for i, el in enumerate(_price_sorted[_counts > 1]):
            _ind_or = np.where(np.isclose(_in_price, el))  # indices of duplicates within original array
            _ind_new = np.where(np.isclose(_price_sorted, el))  # indices of duplicates within new array
            _out_size[_ind_new] = np.sum(_in_size[_ind_or])

        if isBid:
            # reverse arrays to have first element highest bid
            self.data_orderBookBidBars = np.flip(np.transpose([_price_sorted, _out_size]), axis=0)
            if self.data_orderBookBidBars.size > 2 and self.data_orderBookAskBars.size > 2:
                self.data_orderBookAskBars = np.delete(self.data_orderBookAskBars,
                                                       np.where(self.data_orderBookAskBars[:, 0] <= self.data_orderBookBidBars[0, 0]),
                                                       axis=0)
        else:
            self.data_orderBookAskBars = np.transpose([_price_sorted, _out_size])
            if self.data_orderBookBidBars.size > 2 and self.data_orderBookAskBars.size > 2:
                self.data_orderBookBidBars = np.delete(self.data_orderBookBidBars,
                                                       np.where(self.data_orderBookBidBars[:, 0] >= self.data_orderBookAskBars[0, 0]),
                                                       axis=0)

        #print("time process FixDuplicates", round(1e3 * (time.perf_counter() - _start_t), 3))

    def msg_Connected(self):
        """message: received Connected Confirmation queue"""
        self.isConnected = True
        self.ibapi_Status_lineEdit.setText("OK")
        self.ibapi_Status_lineEdit.setStyleSheet("color: %s}" % DGREEN)
        self.ibap_Connect_Button.setChecked(True)
        #self.addtoLogger("Connected to TWS")
        logging.info("Connected to TWS")

    def msg_ConnectionError(self, q: Queue):
        """message: received Connection Error queue"""
        self.isConnected = False
        self.ibapi_Status_lineEdit.setText(f"Er.: {q[1]}")
        self.ibapi_Status_lineEdit.setStyleSheet("color: %s}" % DRED)
        self.ibap_Connect_Button.setChecked(False)
        logging.info(f"Connection error: {q[1]}")

    def msg_Error(self, q: Queue):
        """message: received Other IB API error queue"""
        if q[1] == 200:
            self.ticker_Status_lineEdit.setText("Error: try again")
            self.ticker_Status_lineEdit.setStyleSheet("color: %s}" % DRED)
            self.ticker_Symbol_lineEdit.setStyleSheet("color: %s}" % DRED)

        with open(self.filename_Errors, "a") as myfile:
            myfile.write(datetime.datetime.utcnow().strftime("%H:%M:%S:%f")+" -> "+f"{q[1]}: {q[2]} \n")

        logging.info(f"Error: {q[1]}, {q[2]}")

    def msg_AccountUpdate(self, q: Queue):
        """message: received Account Update queue"""
        self.last_Cash = float(q[2])
        self.totalValue_doubleSpinBox.setValue(q[1])
        self.totalCash_doubleSpinBox.setValue(self.last_Cash)
        self.positionValue_doubleSpinBox.setValue(q[3])
        if q[1] > 0.0:
            self.positionValuePer_doubleSpinBox.setValue(100 * q[3] / q[1])
        _posPLper = 100.*q[4]/(q[1]-q[4]) if q[1] != q[4] else 0
        self.dayPL_doubleSpinBox.setValue(q[4])
        self.dayPL_doubleSpinBox.setStyleSheet("background-color: %s}" % (
            DRED if _posPLper < 0 else DGREY if _posPLper < 2 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
        self.dayPLper_doubleSpinBox.setValue(_posPLper)
        self.dayPLper_doubleSpinBox.setStyleSheet("background-color: %s}" % (
            DRED if _posPLper < 0 else DGREY if _posPLper < 2 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
        self.ibap_Portfolio_Button.setChecked(True)
        self.ibap_Portfolio_Button.setText("Stop")

        self.update_OrderGuiValues()

        #self.addtoLogger("Account Updated")
        logging.debug("Account Updated")

    def msg_PortfolioUpdate(self, q: Queue):
        """message: received Portfolio Update queue"""
        _q = q[1][0:5]
        _lastPosSize = float(_q[0])
        _lastPosPrice = float(_q[1])
        _averagePrice = float(_q[2])
        _positionPL = float(_q[3]) #if _lastPosSize > 0 else 0  # unrealized
        _ticker = str(_q[4])
        if _averagePrice > 0: self.last_avrgSharePrice = _averagePrice

        if self.positionTicker_lineEdit.text() == _ticker or _lastPosSize != 0:  # sell (current) or buy

            self.last_PositionSize = _lastPosSize
            self.positionSize_spinBox.setValue(_lastPosSize)
            self.averagePrice_doubleSpinBox.setValue(_averagePrice)
            self.ref_level2PositionHline.setPos(_averagePrice)

            _PL = self.positionValue_doubleSpinBox.value() - _averagePrice*_lastPosSize
            _posPLper = 100. * (_lastPosPrice / _averagePrice - 1) if _averagePrice != 0 else 0
            _posPLperTotal = _posPLper*self.positionValuePer_doubleSpinBox.value()/100

            self.positionPL_doubleSpinBox.setValue(_positionPL)
            self.positionPL_doubleSpinBox.setStyleSheet("background-color: %s}" % (
                DRED if _posPLper < 0 else DGREY if _posPLper < 1 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
            self.positionPLper_doubleSpinBox.setValue(_posPLper)
            self.positionPLper_doubleSpinBox.setStyleSheet("background-color: %s}" % (
                DRED if _posPLper < 0 else DGREY if _posPLper < 1 else DGREEN if _posPLper < 5 else DBLUE if _posPLper < 10 else PURPLE))
            self.positionPLperTotal_doubleSpinBox.setValue(_posPLperTotal)
            self.positionPLperTotal_doubleSpinBox.setStyleSheet("background-color: %s}" % (
                DRED if _posPLperTotal < 0 else DGREY if _posPLperTotal < 1 else DGREEN if _posPLperTotal < 5 else DBLUE if _posPLperTotal < 10 else PURPLE))
            self.positionTicker_lineEdit.setText(_ticker)

            self.ibap_Portfolio_Button.setChecked(True)
            self.ibap_Portfolio_Button.setText("Stop")

            self.update_OrderGuiValues()

            #self.addtoLogger("Portfolio Updated")
            logging.debug("Portfolio Updated")

    def msg_PositionUpdate(self, q: Queue):
        """message: received Portfolio Update queue"""
        self.last_PositionSize = int(q[1])
        self.positionSize_spinBox.setValue(self.last_PositionSize)
        self.averagePrice_doubleSpinBox.setValue(q[2])
        if q[2] > 0: self.last_avrgSharePrice = q[2]
        self.ref_level2PositionHline.setPos(q[2])
        self.positionTicker_lineEdit.setText(q[3])
        self.ibap_Portfolio_Button.setChecked(True)
        self.ibap_Portfolio_Button.setText("Stop")

        self.update_OrderGuiValues()

        #self.addtoLogger("Position Updated")
        logging.debug("Position Updated")

    def msg_AccPortStoped(self):
        """message: received Stopped Account/Portfolio Confirmation queue"""
        self.isAccountPortfolioUpdating = False
        self.ibap_Portfolio_Button.setChecked(False)
        self.ibap_Portfolio_Button.setText("Update")

        #self.addtoLogger("Account/Portfolio Updates stopped")
        logging.info("Account/Portfolio Updates stopped")

    def msg_CompanyInfoEnd(self, q: Queue):
        """message: received Company Info Received/Ended queue"""
        self.ticker_Status_lineEdit.setText(q[1])
        self.ticker_Status_lineEdit.setStyleSheet("color: %s}" % DGREEN)
        self.ticker_Symbol_lineEdit.setStyleSheet("color: %s}" % DGREEN)

        #self.addtoLogger("Company Info Received/Ended")
        logging.info("Company Info Received/Ended")

    def msg_SlowHistoryEnd(self, q: Queue):
        """message: received Slow History End queue"""
        _candle = np.array(q[1])
        _bar = np.array(q[2])
        _curve = np.array(q[3])

        if self.slow_timeFrame_comboBox.currentText() == "1 day":
            for i, el in enumerate(_candle[:, 0]):
                _unix_date = time.mktime(datetime.datetime.strptime(str(int(el)), "%Y%m%d").timetuple())
                _candle[i, 0] = _unix_date
                _bar[i, 0] = _unix_date
                _curve[i, 0] = _unix_date

        self.data_slowCandles = _candle
        self.data_slowBars = _bar
        self.data_slowCurve = _curve
        if self.historyValidate("TRADES_SLOW"):
            self.update_slow_price_plot(True, fullUpdate=True)
            self.update_slow_volume_plot(True, fullUpdate=True)
            logging.info("Slow History End Received")
        else:
            logging.info("Slow History End Received, but empty")

    def msg_FastHistoryEnd(self, q: Queue):
        """message: received Fast History End queue"""
        _type = q[1]
        _candle = np.array(q[2])

        if _type == "TRADES":
            _bar = np.array(q[3])
            _curve = np.array(q[4])
            self.data_fastCandles = _candle
            self.data_fastCurve = _curve
            self.data_fastBars = _bar
            if self.historyValidate("TRADES_FAST"):
                self.update_fast_price_plot(True, fullUpdate=True)
                self.update_fast_volume_plot(True, fullUpdate=True)
                logging.info(f"Fast History End Received {_type}")
            else:
                logging.info(f"Fast History End Received {_type}, but empty")
        elif _type == "BID":
            self.data_fastCurveBid = _candle
            if self.historyValidate("BID") and self.historyValidate("ASK"):
                self.update_fast_price_plot(True, fullUpdate=True)
                logging.info(f"Fast History End Received {_type}")
            else:
                logging.info(f"Fast History End Received {_type}, but empty")
        elif _type == "ASK":
            self.data_fastCurveAsk = _candle
            if self.historyValidate("BID") and self.historyValidate("ASK"):
                self.update_fast_price_plot(True, fullUpdate=True)
                logging.info(f"Fast History End Received {_type}")
            else:
                logging.info(f"Fast History End Received {_type}, but empty")


    def historyValidate(self, dataType = ""):
        """ check whether data is ok for plotting"""
        _isValid = False
        if dataType == "TRADES_SLOW":
            _isValid = self.data_slowCandles.size > 5 and self.data_slowBars.size > 2 and self.data_slowCurve.size > 2
        elif dataType == "TRADES_FAST":
            _isValid = self.data_fastCandles.size > 5 and self.data_fastBars.size > 2 and self.data_fastCurve.size > 2
        elif dataType == "BID":
            _isValid = self.data_fastCurveBid.size > 2
        elif dataType == "ASK":
            _isValid = self.data_fastCurveAsk.size > 2
        return _isValid

    def msg_SlowHistoryUpdate(self, q: Queue):
        """message: received Slow History Update queue"""
        _candle = np.array(q[1])
        _bar = np.array(q[2])
        _curve = np.array(q[3])

        if self.slow_timeFrame_comboBox.currentText() == "1 day":
            _unix_date = time.mktime(datetime.datetime.strptime(str(int(_candle[0])), "%Y%m%d").timetuple())
            _candle[0] = _unix_date
            _bar[0] = _unix_date
            _curve[0] = _unix_date

        if self.isLevel1Updating and self.historyValidate("TRADES_SLOW"):
            if self.data_slowCandles[-1, 0] == _candle[0]:  # update
                self.data_slowCandles[-1, :] = _candle
            else:  # or add
                self.data_slowCandles = np.vstack((self.data_slowCandles, _candle))

            if self.data_slowBars[-1, 0] == _bar[0]:  # update
                self.data_slowBars[-1, :] = _bar
            else:  # or add
                self.data_slowBars = np.vstack((self.data_slowBars, _bar))

            if _curve[1] < 0:  # trades not valid= take previous as temporary
                _curve = np.array([_curve[0], self.data_slowCurve[-1, 1]])

            if self.data_slowCurve[-1, 0] == _curve[0]:  # update
                self.data_slowCurve[-1, :] = _curve
            else:  # or add
                self.data_slowCurve = np.vstack((self.data_slowCurve, _curve))

            self.update_slow_price_plot()
            self.update_slow_volume_plot()
            logging.debug("Slow History Updated")
    def msg_HighLowUpdate(self, q: Queue):
        """message: received High/Low History Update queue"""
        if self.isLevel1Updating:
            self.data_1min_HighLows = np.array(q[1])
            self.update_HighLow_Values()

            self.closeChange_indicator_lineEdit.setStyleSheet("background-color: %s}" % (DRED if q[2] else DGREEN))  # if close up = dip buy/green

            logging.debug("High/Low Updated")

    def msg_FastHistoryUpdate(self, q: Queue):
        """message: received Fast History Update queue"""
        if self.isLevel1Updating:
            _type = q[1]
            _candle = np.array(q[2])
            if _type == "TRADES" and self.historyValidate("TRADES_FAST"):
                _bar = np.array(q[3])
                _curve = np.array(q[4])
                if self.data_fastCandles[-1, 0] == _candle[0]:  # update
                    self.data_fastCandles[-1, :] = _candle
                elif self.data_fastCandles[-1, 0] <= _candle[0]:  # or add
                    self.data_fastCandles = np.vstack((self.data_fastCandles, _candle))
                if self.data_fastBars[-1, 0] == _bar[0]:  # update
                    self.data_fastBars[-1, :] = _bar
                elif self.data_fastBars[-1, 0] <= _bar[0]:  # or add
                    self.data_fastBars = np.vstack((self.data_fastBars, _bar))

                if _curve[1] < 0:  # trades not valid= take previous as temporary
                    _curve = np.array([_curve[0], self.data_fastCurve[-1, 1]])

                if self.data_fastCurve[-1, 0] == _curve[0]:  # update
                    self.data_fastCurve[-1, :] = _curve
                elif self.data_fastCurve[-1, 0] <= _curve[0]:  # or add
                    self.data_fastCurve = np.vstack((self.data_fastCurve, _curve))

            elif _type == "BID" and self.historyValidate("BID"):
                if self.data_fastCurveBid[-1, 0] == _candle[0]:  # update
                    self.data_fastCurveBid[-1, :] = _candle
                elif self.data_fastCurveBid[-1, 0] <= _candle[0]:  # or add
                    self.data_fastCurveBid = np.vstack((self.data_fastCurveBid, _candle))
            elif _type == "ASK" and self.historyValidate("ASK"):
                if self.data_fastCurveAsk[-1, 0] == _candle[0]:  # update
                    self.data_fastCurveAsk[-1, :] = _candle
                elif self.data_fastCurveAsk[-1, 0] <= _candle[0]:  # or add
                    self.data_fastCurveAsk = np.vstack((self.data_fastCurveAsk, _candle))

            if self.historyValidate("TRADES_FAST") and self.historyValidate("BID") and self.historyValidate("ASK"):
                self.update_fast_price_plot()
                self.update_fast_volume_plot()

            logging.debug(f"Fast History Updated {_type}")

    def msg_Level1Update(self, q: Queue):
        """message: received Level1 Update queue"""
        if self.isLevel1Updating:
            _type = q[1][0]
            _value = q[1][1]
            if _type == 1:  # BID Price
                self.last_Bid = _value
                self.last_MidPoint = (self.last_Bid + self.last_Ask)/2
                self.update_OrderGuiValues()

                self.last_Spread = 100*(2*(self.last_Ask-self.last_Bid)/(self.last_Ask+self.last_Bid))
                self.ticker_spreadper_doubleSpinBox.setValue(self.last_Spread)
            elif _type == 2:    # ASK Price
                self.last_Ask = _value
                self.last_MidPoint = (self.last_Bid + self.last_Ask) / 2
                self.update_OrderGuiValues()

                self.last_Spread = 100 * (2 * (self.last_Ask - self.last_Bid) / (self.last_Ask + self.last_Bid))
                self.ticker_spreadper_doubleSpinBox.setValue(self.last_Spread)
            elif _type == 4:    # LAST Price
                # if self.last_Price < _value:
                #     self.ref_orderBookHlineLast.setPen("g")
                #     self.ref_slowCandlesHlineLast.setPen("g")
                # elif self.last_Price > _value:
                #     self.ref_orderBookHlineLast.setPen("r")
                #     self.ref_slowCandlesHlineLast.setPen("r")
                # self.ref_orderBookHlineLast.setPos(_value)
                # self.ref_slowCandlesHlineLast.setPos(_value)

                self.last_Price = _value
                self.ticker_lastPrice_doubleSpinBox.setValue(_value)
                self.last_Change = 100*((self.last_Price/self.last_Close)-1.)
                self.ticker_dayChangeper_doubleSpinBox.setValue(self.last_Change)
                self.ticker_lastPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Change >= 0 else DRED))
                self.ticker_dayChangeper_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Change >= 0 else DRED))
                self.update_OrderGuiValues()

            elif _type == 6:    # HIGH
                self.last_High = _value
                self.ticker_highDayPrice_doubleSpinBox.setValue(_value)
                self.ticker_highDayPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_High >= self.last_Close else DRED))
            elif _type == 7:    # LOW
                self.last_Low = _value
                self.ticker_lowDayPrice_doubleSpinBox.setValue(_value)
                self.ticker_lowDayPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Low >= self.last_Close else DRED))
            elif _type == 9:    # CLOSE price
                self.last_Close = _value  # of the previous day
                self.last_Change = 100*((self.last_Price/self.last_Close)-1.)
                self.ticker_dayChangeper_doubleSpinBox.setValue(self.last_Change)
                self.ticker_lastPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Change >= 0 else DRED))
                self.ticker_dayChangeper_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Change >= 0 else DRED))
                self.ticker_highDayPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_High >= self.last_Close else DRED))
                self.ticker_lowDayPrice_doubleSpinBox.setStyleSheet(
                    "color: %s}" % (DGREEN if self.last_Low >= self.last_Close else DRED))
            elif _type == 5:    # LAST Size i.e. volume of last trade
                self.last_Size = _value
            elif _type == 8:    # Volume of the day (in millions)
                self.last_Volume = _value/10000
                self.ticker_dayVolume_doubleSpinBox.setValue(self.last_Volume)

                self.ticker_relativeVolume_doubleSpinBox.setValue(
                    self.last_Volume / self.last_avrg_Volume if self.last_avrg_Volume > 1e-9 else 0.)
            elif _type == 21:   # Volume over 90 day (in millions)
                self.last_avrg_Volume = _value/10000
                self.ticker_relativeVolume_doubleSpinBox.setValue(
                    self.last_Volume / self.last_avrg_Volume if self.last_avrg_Volume > 1e-9 else 0.)

            elif _type == 56:   # Volume per minute in thousands
                self.last_VolumeRate = _value/1000
                self.ticker_rateVolume_doubleSpinBox.setValue(self.last_VolumeRate)

            if self.ticker_Symbol_lineEdit.text().upper() == self.positionTicker_lineEdit.text().upper():
                self.update_PositionGuiValue()
            #self.addtoLogger(f"Level1 Updated: type:{_type}, value: {_value}")
            logging.debug(f"Level1 Updated: type:{_type}, value: {_value}")

    def msg_ML_update(self, q: Queue):
        if self.isLevel1Updating:
            # probab = q[1][0, 1]  # "Confidence"].[-2], "Confidence".[-1]
            # summary = q[1][2-5]  # "precision_Pos", "recall_Pos", "precision_Neg", "recall_Neg"
            # cum_eff = q[1][6-8]  # "cum_predicted", "cum_perfect"

            self.ml_probaL_spinBox.setValue(int(q[1][0]))
            self.ml_probaC_spinBox.setValue(int(q[1][1]))
            self.ml_probaL_spinBox.setStyleSheet("background-color: %s}" % (DRED if int(q[1][0]) < 50 else DGREEN))
            self.ml_probaC_spinBox.setStyleSheet("background-color: %s}" % (DRED if int(q[1][1]) < 50 else DGREEN))

            self.ml_precisionBuy_spinBox.setValue(int(q[1][2]))
            self.ml_recallBuy_spinBox.setValue(int(q[1][3]))
            self.ml_precisionSell_spinBox.setValue(int(q[1][4]))
            self.ml_recallSell_spinBox.setValue(int(q[1][5]))

            self.ml_cumPredicted_spinBox.setValue(int(q[1][6]))
            self.ml_cumPerfect_spinBox.setValue(int(q[1][7]))

            self.fps_MLEvents_doubleSpinBox.setValue(int(q[1][8]))

    def msg_Level2Update(self, q: Queue):
        """message: received Level2 Update queue"""
        if self.isLevel2Updating:
            if q[1] == "TRADES":  # updats tick based arrays
                _in_ticks = np.array(q[2])
                self.data_volumeProf_update_size += _in_ticks.shape[0]
                self.data_level2Scatter_update_size += _in_ticks.shape[0]

                if self.data_level2Scatter_ticks.size == 0:
                    self.data_level2Scatter_ticks = _in_ticks
                else:
                    self.data_level2Scatter_ticks = np.vstack((self.data_level2Scatter_ticks,
                                                               _in_ticks))

                if self.data_level2Scatter_ticks.size > 3:
                    self.last_Price = self.data_level2Scatter_ticks[-1, 1]
                    self.last_Size = self.data_level2Scatter_ticks[-1, 2]
                elif self.data_level2Scatter_ticks.size > 0:
                    self.last_Price = self.data_level2Scatter_ticks[1]
                    self.last_Size = self.data_level2Scatter_ticks[2]

                self.orderTape_lineEdit.setText(str(self.last_Price) + " -> " + str(int(self.last_Size)))
                self.ref_orderBookHlineLast.setPos(self.last_Price)
                self.ref_orderBookHlineLast_P1per.setPos(self.last_Price*1.01)
                self.ref_orderBookHlineLast_M1per.setPos(self.last_Price*0.99)
                self.ref_orderBookHlineLast_P3per.setPos(self.last_Price*1.03)
                self.ref_orderBookHlineLast_M3per.setPos(self.last_Price*0.97)
                self.ref_slowCandlesHlineLast.setPos(self.last_Price)

                if self.last_Price >= self.last_Ask:
                    self.ref_orderBookHlineLast.setPen(pg.mkPen(DGREEN, width=2))
                    self.ref_slowCandlesHlineLast.setPen("g")
                    self.orderTape_lineEdit.setStyleSheet("background-color: %s}" % DGREEN)
                elif self.last_Price <= self.last_Bid:
                    self.ref_orderBookHlineLast.setPen(pg.mkPen(DRED, width=2))
                    self.ref_slowCandlesHlineLast.setPen("r")
                    self.orderTape_lineEdit.setStyleSheet("background-color: %s}" % DRED)
                else:
                    self.ref_orderBookHlineLast.setPen(pg.mkPen(YELLOW, width=2))
                    self.ref_slowCandlesHlineLast.setPen("y")
                    self.orderTape_lineEdit.setStyleSheet("background-color: %s}" % DBLUE)

                if self.isLevel1Updating:
                    self.update_HighLow_Values()

                if self.saveTrades_Button.isChecked():
                    with open(self.filename_Level2Trades, "a") as f:
                        np.savetxt(f, _in_ticks if len(_in_ticks.shape) > 1 else _in_ticks.reshape((1, -1)), fmt='%.2f',
                                   delimiter=" ")
                if self.ticker_Symbol_lineEdit.text().upper() == self.positionTicker_lineEdit.text().upper():
                    self.update_PositionGuiValue()

                self.update_OrderGuiValues()
                self.recalculate_Level2_Trades()
                self.recalculate_Level2_cumulDelta()
                self.recalculate_VolumeProfile(fullUpdate=False, isSum=self.sumVolProf_checkBox.isChecked())

            elif q[1] == "BA":  # updats tick based arrays
                _in_ticks = np.array(q[2])  # [time, bidPrice, askPrice, bidSize, askSize]

                if self.data_level2BACurves_ticks.size == 0:
                    self.data_level2BACurves_ticks = _in_ticks
                else:
                    self.data_level2BACurves_ticks = np.vstack((self.data_level2BACurves_ticks, _in_ticks))

                # _bid = self.last_Bid
                # _ask = self.last_Ask

                if self.data_level2BACurves_ticks.size > 5:  # more than one element/enough to calculate
                    self.last_Bid = self.data_level2BACurves_ticks[-1, 1]
                    self.last_Ask = self.data_level2BACurves_ticks[-1, 2]
                elif self.data_level2BACurves_ticks.size > 0:
                    self.last_Bid = self.data_level2BACurves_ticks[1]  # todo fix index 1 is out of bounds for axis 0 with size 1
                    self.last_Ask = self.data_level2BACurves_ticks[2]

                self.recalculate_Level2_BA()  # also will update the plot
                #self.recalculate_Level2_sizeBA()  # also will update the plot

                self.last_MidPoint = (self.last_Bid + self.last_Ask) / 2
                self.update_OrderGuiValues()

                if self.saveBidAsks_Button.isChecked():
                    with open(self.filename_Level2BA, "a") as f:
                        np.savetxt(f, _in_ticks if len(_in_ticks.shape) > 1 else _in_ticks.reshape((1, -1)), fmt='%.2f', delimiter=" ")

            elif q[1] == "ORDERBOOK_BID":  # update order book bid array
                self.data_orderBookBidBars = q[2]  # volume in x100
                self.remove_dupes_BidAsk()

                # if more than one bar  = worth using as an update and calculations
                if self.data_orderBookBidBars.size > 2:
                    self.data_orderBook_update_size += 1  # counter for the image resampling

                    self.last_sumBidSize = np.sum(self.data_orderBookBidBars[:, 1])  # negative
                    self.last_sumPriceBidSize = np.sum(self.data_orderBookBidBars[:, 0]*self.data_orderBookBidBars[:, 1])  # negative
                    self.last_avrgBidPrice = self.last_sumPriceBidSize/self.last_sumBidSize  # positive

                    _dec = self.priceMinIncrement
                    _inv_distance = np.power(
                        np.abs(self.data_orderBookBidBars[:, 0] - self.data_orderBookBidBars[0, 0]) + _dec, -1)
                    self.last_forceBid = np.sum(self.data_orderBookBidBars[:, 1]*_inv_distance)  # negative

                    self.data_level2Image_update = self.data_orderBookBidBars
                    if self.data_level2Image_TimeScale_ticks.size == 0:
                        self.data_level2Image_TimeScale_ticks = np.array([time.time()])
                    else:
                        self.data_level2Image_TimeScale_ticks = np.append(self.data_level2Image_TimeScale_ticks,
                                                                          time.time())
                    self.recalculate_Map(self.update_Map_ticks())
                    self.update_order_book_plot()
                    self.recalculate_sumBidAsk(time.time())  # recallculate Level2 sumBidAsk with latest sum Bid at given time

                if self.saveOrderBook_Button.isChecked():
                    with open(self.filename_Level2OrderBook, "a") as f:
                        np.savetxt(f, -self.data_orderBookBidBars.T, fmt='%.2f', delimiter=' ')

            elif q[1] == "ORDERBOOK_ASK":  # update order book bid array
                self.data_orderBookAskBars = q[2]  # volume in x100
                self.remove_dupes_BidAsk(False)

                # if more than one bar  = worth using as an update and calculations
                if self.data_orderBookAskBars.size > 2:
                    self.data_orderBook_update_size += 1  # counter for the image resampling

                    self.last_sumAskSize = np.sum(self.data_orderBookAskBars[:, 1])
                    self.last_sumPriceAskSize = np.sum(self.data_orderBookAskBars[:, 0]*self.data_orderBookAskBars[:, 1])
                    self.last_avrgAskPrice = self.last_sumPriceAskSize/self.last_sumAskSize

                    _dec = self.priceMinIncrement
                    _inv_distance = np.power(
                        np.abs(self.data_orderBookAskBars[:, 0] - self.data_orderBookAskBars[0, 0]) + _dec, -1)
                    self.last_forceAsk = np.sum(self.data_orderBookAskBars[:, 1] * _inv_distance)

                    self.data_level2Image_update = self.data_orderBookAskBars
                    if self.data_level2Image_TimeScale_ticks.size == 0:
                        self.data_level2Image_TimeScale_ticks = np.array([time.time()])
                    else:
                        self.data_level2Image_TimeScale_ticks = np.append(self.data_level2Image_TimeScale_ticks,
                                                                          time.time())

                    self.recalculate_Map(self.update_Map_ticks())
                    self.update_order_book_plot()
                    self.recalculate_sumBidAsk(time.time())  # recallculate Level2 sumBidAsk with latest sum Bid at given time

                if self.saveOrderBook_Button.isChecked():
                    with open(self.filename_Level2OrderBook, "a") as f:
                        np.savetxt(f, self.data_orderBookAskBars.T, fmt='%.2f', delimiter=' ')

            logging.debug(f"Level2 Updated: type:{q[1]}, value: {q[2]}")

    def msg_ScannerUpdate(self, q: Queue, isFull):
        """message: received Scanner data/list queue"""
        _LIMIT = 10  # HoD/LoD/VWAP levels proximity limit
        _valid_ID = q[1] != "*"
        _names = q[1][_valid_ID]

        self.scannerList_tableWidget.setSortingEnabled(False)

        if isFull:
            # 0 last price/ 1 Volume of the day/2 close price of the precious day/3 bid/4 ask/avrg_volume
            _data = q[2][_valid_ID, :]
            self.scannerList_tableWidget.clearContents()
            self.scannerList_tableWidget.setRowCount(len(_names))

            for _row, _name in enumerate(_names):
                self.scannerList_tableWidget.setItem(_row, 0, QtWidgets.QTableWidgetItem(_name))
                _last = _data[_row, 0]
                _change = (int(1000 * (_data[_row, 0] / _data[_row, 2] - 1)) / 10) if _data[_row, 2] > 0.01 else 0.
                _volume = round(0.0001*_data[_row, 1], 2)
                _rel_volume = (int(100 * _data[_row, 1] / _data[_row, 5]) / 100) if _data[_row, 5] > 1 else 0.
                _spread = (int(10000 * (_data[_row, 4] / _data[_row, 3] - 1)) / 100) if _data[_row, 3] > 0.01 else 0.
                _HoD = int(1000 * (_data[_row, 6] / _data[_row, 0] - 1)) / 10  # negative = above HoD
                _LoD = -int(1000 * (_data[_row, 7] / _data[_row, 0] - 1)) / 10  # negative = below LoD
                _VWAP = int(1000 * (_data[_row, 8] / _data[_row, 0] - 1)) / 10  # positive/negative = below/above VWAP

                for _col in range(9):
                    _item = QtWidgets.QTableWidgetItem()
                    if _col == 0:
                        _item.setData(QtCore.Qt.EditRole, int(_row+1))
                    elif _col == 1:
                        _item.setData(QtCore.Qt.EditRole, float(_last))
                    elif _col == 2:
                        _item.setData(QtCore.Qt.EditRole, float(_change))
                        _item.setBackground(QtGui.QColor(DRED if _change < 0 else DGREEN))
                    elif _col == 3:
                        _item.setData(QtCore.Qt.EditRole, float(_spread))
                    elif _col == 4:
                        _item.setData(QtCore.Qt.EditRole, float(_volume))
                    elif _col == 5:
                        _item.setData(QtCore.Qt.EditRole, float(_rel_volume))
                    elif _col == 6:
                        _item.setData(QtCore.Qt.EditRole, float(_HoD))
                        _item.setBackground(QtGui.QColor(DGREEN if _LIMIT > _HoD else BLACK))
                    elif _col == 7:
                        _item.setData(QtCore.Qt.EditRole, float(_LoD))
                        _item.setBackground(QtGui.QColor(DGREEN if _LIMIT > _LoD else BLACK))
                    elif _col == 8:
                        _item.setData(QtCore.Qt.EditRole, float(_VWAP))
                        _item.setBackground(
                            QtGui.QColor(DRED if _LIMIT > _VWAP > 0 else DGREEN if 0 > _VWAP > -_LIMIT else BLACK))
                    self.scannerList_tableWidget.setItem(_row, _col + 1, _item)
            #print("Scanner ticker Full update")
        else:
            # 0 last price/ 1 Volume of the day/2 close price of the precious day/3 bid/4 ask/avrg_volume
            _data = q[2]
            if self.scannerList_tableWidget.rowCount() >= 1:
                pass
            else:
                self.scannerList_tableWidget.setRowCount(len(_names))
                for _row, _name in enumerate(_names):
                    self.scannerList_tableWidget.setItem(_row, 0, QtWidgets.QTableWidgetItem(_name))

            _last = _data[0]
            _change = (int(1000 * (_data[0] / _data[2] - 1)) / 10) if _data[2] > 0.01 else 0.
            _volume = round(0.0001*_data[1], 2)
            _rel_volume = (int(100 * _data[1] / _data[5]) / 100) if _data[5] > 1 else 0.
            _spread = (int(10000 * (_data[4] / _data[3] - 1)) / 100) if _data[3] > 0.01 else 0.
            _name = q[3]
            _HoD = int(1000 * (_data[6]/_data[0]-1))/10
            _LoD = -int(1000 * (_data[7]/_data[0]-1))/10
            _VWAP = int(1000 * (_data[8]/_data[0]-1))/10
            #print("Scanner ticker Partial update")
            for _row in range(len(_names)):
                if self.scannerList_tableWidget.item(_row, 0).text() == _name:
                #if self.scannerList_tableWidget.cellWidget(el, 0).currentText() == _name:
                    _item_last = QtWidgets.QTableWidgetItem()
                    _item_last.setData(QtCore.Qt.EditRole, float(_last))
                    self.scannerList_tableWidget.setItem(_row, 2, _item_last)

                    _item_change = QtWidgets.QTableWidgetItem()
                    _item_change.setData(QtCore.Qt.EditRole, float(_change))
                    _item_change.setForeground(QtGui.QColor(DRED if _change < 0 else DGREEN))
                    self.scannerList_tableWidget.setItem(_row, 3, _item_change)

                    _item_spread = QtWidgets.QTableWidgetItem()
                    _item_spread.setData(QtCore.Qt.EditRole, float(_spread))
                    self.scannerList_tableWidget.setItem(_row, 4, _item_spread)

                    _item_volume = QtWidgets.QTableWidgetItem()
                    _item_volume.setData(QtCore.Qt.EditRole, float(_volume))
                    self.scannerList_tableWidget.setItem(_row, 5, _item_volume)

                    _item_rel_volume = QtWidgets.QTableWidgetItem()
                    _item_rel_volume.setData(QtCore.Qt.EditRole, float(_rel_volume))
                    self.scannerList_tableWidget.setItem(_row, 6, _item_rel_volume)

                    _item_HoD = QtWidgets.QTableWidgetItem()
                    _item_HoD.setData(QtCore.Qt.EditRole, float(_HoD))
                    self.scannerList_tableWidget.setItem(_row, 7, _item_HoD)

                    _item_LoD = QtWidgets.QTableWidgetItem()
                    _item_LoD.setData(QtCore.Qt.EditRole, float(_LoD))
                    self.scannerList_tableWidget.setItem(_row, 8, _item_LoD)

                    _item_VWAP = QtWidgets.QTableWidgetItem()
                    _item_VWAP.setData(QtCore.Qt.EditRole, float(_VWAP))
                    self.scannerList_tableWidget.setItem(_row, 9, _item_VWAP)

                    break

        self.scannerList_tableWidget.resizeColumnsToContents()
        self.scannerList_tableWidget.setSortingEnabled(True)

        #self.addtoLogger("Scanner Data Received")
        logging.debug("Scanner Data Received")

    def msg_NewsUpdate(self, q: Queue):
        """message: received Scanner data/list queue"""
        _msg = q[1]
        _text = ""
        for el in _msg:
            _text += el+"\n\n"
        self.news_textBrowser.setText(_text)

        #self.addtoLogger("News Data Received")
        logging.info("News Data Received")

    def msg_OrderInfo(self, q: Queue):  # todo fix sometimes double sell confirmation for stoploss bracket
        """message: received Scanner data/list queue"""
        _type = q[1]
        _data = q[2]

        def tup_same(in1, in2):
            _same = False
            if in1 is None or in2 is None:
                pass
            elif len(in1) == len(in2):
                _count = 0
                for i in range(len(in1)):
                    if in1[i] == in2[i]:
                        _count +=1
                if _count == len(in1):
                    _same = True
            return _same

        if not tup_same(self.last_OrderOpen, _data) and not tup_same(self.last_OrderStatus, _data):
            if _type == "OPEN":
                # _data = (" OrderId:", orderId, "Status:", orderState.status,  # 1,3
                #              "Symbol:", contract.symbol, "Action:", order.action,  # 5,7
                #              "OrderType:", order.orderType, "TotalQty:", order.totalQuantity,  # 9, 11
                #              "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice)  # 13, 15

                self.orderStatus_lineEdit.setText(f"{_data[3]}-{_data[1]}:{_data[5]} {_data[7]} {_data[9]} {_data[11]}@{_data[13]}")
                self.last_OrderOpen = _data

                if _data[3] == "Filled":  # add to the transaction tape
                    if len(self.data_OrderTape_Ticker_Side) == 0:
                        self.data_OrderTape_Ticker_Side = [[_data[5], _data[7]]]
                    else:
                        self.data_OrderTape_Ticker_Side.append([_data[5], _data[7]])

            elif _type == "STATUS":
                # _data = ("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled, "Remaining:", remaining,  # 1,3,5,7
                #              "AvgFillPrice:", avgFillPrice, "LastFillPrice:", lastFillPrice)  # 9,11
                self.orderStatus_lineEdit.setText(f"{_data[3]}-{_data[1]}:Fill-{_data[5]}/Remain-{_data[7]}@{_data[9]}")
                self.last_OrderStatus = _data

                if self.last_OrderOpen is not None:
                    _isBuy = self.last_OrderOpen[7] == "BUY"
                    if _data[3] == "Filled":  #add to the transaction tape
                        _ar = np.array([_data[1],
                                        _data[5]*(1 if _isBuy else -1),
                                        _data[9]])  # id*Size(+Buy/-Sell)*Price
                        if self.data_OrderTape.size == 0:
                            self.data_OrderTape = _ar
                        else:
                            self.data_OrderTape = np.vstack((self.data_OrderTape, _ar))

                        if _isBuy:
                            _text = f"{self.data_OrderTape_Ticker_Side[-1][1]}-{self.data_OrderTape_Ticker_Side[-1][0]}: {int(_ar[1])} @ {_ar[2]}"
                            self.addtoLogger(_text, DBLUE, WHITE)
                        else:  # sell
                            _profit = round(abs(_ar[1]) * (_ar[2] - self.last_avrgSharePrice), 2)
                            _per = round(100*((_ar[2]/self.last_avrgSharePrice)-1), 2)

                            _text = f"{self.data_OrderTape_Ticker_Side[-1][1]}-{self.data_OrderTape_Ticker_Side[-1][0]}: {int(_ar[1])} @ {_ar[2]} => {_profit}({_per}%)"
                            self.addtoLogger(_text, RED_BUTTON if _profit < 0 else GREEN_BUTTON, WHITE)
                            # print("order type", _type)
                            # print("order data", _data)

            # elif _type == "COMPLETED":
            #     self.orderStatus_lineEdit.setText(f"Complete:{_data[3]}-{_data[1]}:{_data[5]} Total-{_data[7]}/Fill-{_data[9]}@{_data[11]}")
            #     # _data = ("CompletedOrder. PermId:", order.permId,
            # #              "Symbol:", contract.symbol, "Action:", order.action, "OrderType:", order.orderType, #1,3,5
            # #              "TotalQty:", order.totalQuantity, "FilledQty:", order.filledQuantity,  # 7,9
            # #              "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status, #11,13,15
            # #              "Completed time:", orderState.completedTime, "Completed Status:" + orderState.completedStatus)
            elif _type == "EXECDETAILS":
                # _data = ("ExecDetails. ReqId:", reqId, "Symbol:", contract.symbol,
                #      "Side:", execution.side, "Number of Shares", execution.shares, "Price:", execution.price)
                _text = f"{_data[5]}-{_data[3]}: {int(_data[7])} @ {_data[9]}"
                self.addtoLogger(_text, BLACK, WHITE)

            with open(self.filename_OrderTape, "a") as myfile:
                myfile.write(datetime.datetime.utcnow().strftime("%H:%M:%S:%f")+" -> "+f"{_type}: {_data} \n")
            logging.info("Order Info Received")

    def closeEvent(self, event):
        """send close event message to main process and IB API via queue"""
        logger.info("Closing GUI")
        #self.addtoLogger("Closing GUI")
        self.outQ.put_nowait(("EXIT",))
        self.exitReq = True
        super().closeEvent(event)



def set_gui_colors(app):
    """sets GUI color scheme"""

    style_list = ['bb10dark', 'bb10bright', 'cleanlooks', 'cde', 'motif', 'plastique', 'Windows', 'Fusion']
    app.setStyle(style_list[7])
    #dark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    #dark_stylesheet = qdarkstyle.load_stylesheet(qt_api=os.environ('PYQTGRAPH_QT_LIB'))
    #app.setStyle(dark_stylesheet)
    
    palette = QtGui.QPalette()

    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    app.setPalette(palette)


def main():
    # app = QtWidgets.QApplication([])  # creates application
    # app.setWindowIcon(QtGui.QIcon('/icons/burn.png'))
    # set_gui_colors(app)
    #
    # GuiWindow()  # creates main window based on ui file
    #
    # app.exec()  # executes/exit gui application
    pass


if __name__ == "__main__":
    main()
