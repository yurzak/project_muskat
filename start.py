# -*- coding: utf-8 -*-
"""
Main file with Creation of Gui and Interactive Brokers Application, Wrapper/Ciient processes
"""

import os, sys, time, datetime
import logging
from multiprocessing import Process, Queue
from PyQt5 import QtWidgets, QtGui, QtCore

from gui import scalpGui
from ib import ibApp
from machina import ml_predict


def setupLogger(file="ScalpRay", levelFile=logging.INFO, levelConsole=logging.WARNING):
    if not os.path.exists("log"):
        os.makedirs("log")

    recfmt = '%(threadName)s@%(processName)s %(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s'
    timefmt = '%y%m%d_%H:%M:%S'

    # logging.basicConfig( level=logging.DEBUG,
    #                    format=recfmt, datefmt=timefmt)
    logging.basicConfig(filename=time.strftime("log/" + file + "_%y%m%d_%H%M%S.log"),
                        filemode="w",
                        level=levelFile,
                        format=recfmt, datefmt=timefmt)
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(levelConsole)
    logger.addHandler(console)


def CreateGui(inQ, inL1Q, inL2Q, inL3Q, outQ):
    # TODO needs permanent fix (current gives blurred image in single high dpi monitor)
    #QtWidgets.QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    setupLogger("GUI", levelFile=logging.INFO, levelConsole=logging.WARNING)
    gui_app = QtWidgets.QApplication([])  # creates application

    gui_app.setWindowIcon(QtGui.QIcon("gui/icons/burn.png"))  # set icon
    scalpGui.set_gui_colors(gui_app)  # set color palette for the application

    # creates main window/gui object based on ui file and passes mainQ(ueue)
    logging.debug("Creating GUI window")
    gui = scalpGui.GuiWindow(inQ, inL1Q, inL2Q, inL3Q, outQ)
    logging.debug("Running Gui App")
    gui_app.exec()  # executes gui application = start gui event loop/process
    logging.debug("Finished Gui App")


def StartIBAPI(inQ, outQ, outL1Q, outL2Q, outL3Q):

    setupLogger("IBapi", levelFile=logging.INFO, levelConsole=logging.WARNING)
    logging.debug("Creating IB API App")
    app = ibApp.IbApp(inQ, outQ, outL1Q, outL2Q, outL3Q)
    logging.debug("Running IB API App")
    app.start()  # start IB app event loop/process
    logging.debug("Finished IB API App")

def StartMLCore(inQ, outQ):

    setupLogger("MLCore", levelFile=logging.INFO, levelConsole=logging.WARNING)
    logging.debug("Creating MLCore")
    ml_core = ml_predict.MachineCore(inQ, outQ)
    logging.debug("Starting MLCore")
    ml_core.start()  # start MLCore event loop/process
    logging.debug("Finished MLCore")

def main():

    setupLogger()
    logging.info("now is %s", datetime.datetime.now())

    gui_inQ = Queue()  # gui_OUT => ib_IN and ib_OUT => gui_IN
    gui_inL1Q = Queue()
    gui_inL2Q = Queue()
    gui_inMLQ = Queue()
    ib_inQ = Queue()
    ml_inQ = Queue()

    # GUI SECTION START
    logging.info("Starting GUI process")
    gui_p = Process(target=CreateGui, args=(gui_inQ, gui_inL1Q, gui_inL2Q, gui_inMLQ, ib_inQ))
    #gui_p.daemon = True
    gui_p.start()  # executes/exit gui application = start gui event loop/thread/process
    logging.info("GUI process started")
    # GUI SECTION END

    ## IB_API SECTION START
    logging.info("Starting IB API process")
    ib_p = Process(target=StartIBAPI, args=(ib_inQ, gui_inQ, gui_inL1Q, gui_inL2Q, ml_inQ))
    ib_p.daemon = True
    ib_p.start()  # start IB app event loop/thread/process
    logging.info("IB API process started")
    ## IB_API SECTION END

    ## MLCORE SECTION START
    logging.info("Starting MLCore process")
    ml_p = Process(target=StartMLCore, args=(ml_inQ, gui_inMLQ))
    ml_p.daemon = True
    ml_p.start()  # start MLcore event loop/thread/process
    logging.info("MLCore process started")
    ## MLCore SECTION END

    gui_p.join()
    logging.info("GUI process finished")
    ib_p.join()
    logging.info("IB_app process finished")
    ml_p.join()
    logging.info("MLCore process finished")


if __name__ == "__main__":
    sys.exit(main())
