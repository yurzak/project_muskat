# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:33:25 2020

@author: DocZhi
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


_color_up = "#26a69a"
_color_down = "#ef5350"
_color_neutral = "#fff176"

# _color_up = "g"
# _color_down = "r"

## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)
class CandlestickItem(pg.GraphicsObject):
    """
    Custom CandleStcik Item
    Data must have fields: time, open, close, min, max

    """
  
    def __init__(self, data, HA=False, time_offset=0):  # offset UTC
        super().__init__()
          # data must have fields: time, open, close, min, max
        if HA:
            self.data = data
            self.convert_to_HeikinAshi()
        else:
            self.data = data[1:, :]
        self.offset = time_offset
        self.picture = None
        self.isHA = HA  # Hiekih-Ashi candles
        self.generatePicture()

    def convert_to_HeikinAshi(self):
        _close = 0.25*np.sum(self.data[1:, 1:], axis=1)
        _open = 0.5*(self.data[:-1, 1]+self.data[:-1, 2])
        _max = np.max(self.data[1:, (1, 2, 4)], axis=1)
        _min = np.min(self.data[1:, (1, 2, 3)], axis=1)
        self.data = np.transpose([self.data[1:, 0], _open[:], _close[:], _min[:], _max[:]])

    def generatePicture(self):
        # pre-computing a QPicture object allows paint() to run much more quickly,
        # rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = (self.data[1, 0] - self.data[0, 0]) / 3.
        for t, open, close, min, max in self.data:

            _color = _color_up if open < close else _color_neutral if open == close else _color_down
            p.setBrush(pg.mkBrush(_color))
            p.setPen(pg.mkPen(_color))

            if min != max:
                p.drawLine(QtCore.QPointF(t+self.offset, min), QtCore.QPointF(t+self.offset, max))
            p.drawRect(QtCore.QRectF(t+self.offset-w, open, w*2, close-open))
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        if self.picture is None:
            self.generatePicture()
        return QtCore.QRectF(self.picture.boundingRect())

    def setData(self, data, HA=False, time_offset=0):
        self.offset = time_offset
        if HA:
            self.data = data
            self.convert_to_HeikinAshi()
        else:
            self.data = data[1:, :]
        self.picture = None
        self.update()
        self.informViewBoundsChanged()

class PercentAxisItem(pg.AxisItem):
    def __init__(self, zero_point, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_point = zero_point

    def tickStrings(self, values, scale, spacing):
        _values = []
        for value in values:
            #_values.append(-100 * (value * len(values) / sum(values) - 1))
            _values.append(round(-100*(value/self.zero_point-1), 3))

        return super().tickStrings(_values, scale, spacing)

