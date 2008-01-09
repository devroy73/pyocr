# -*- coding: iso-8859-15 -*-
# generated by wxGlade 0.6.2 on Wed Jan 09 23:58:32 2008

import wx

# begin wxGlade: dependencies
# end wxGlade

# begin wxGlade: extracode

# end wxGlade

#Exceptions specification
class MultipleFilesException(Exception):
    pass

## Description: Converts RGB data loaded from a bitmap
##              into binary data suitable for the OCR algorithm.
## Parameters: [in] rgb_data RGB data retrieved from the bitmap.
## Return: Binary data array.
def ConvertRgb2BinData(rgb_data):
    bin_data = []
    # 3 - because each pixel has 3 bytes for R,G & B - whereas we
    # need only one indicating 1 or 0
    for i in xrange(len(rgb_data)/3):
        if WHITE == rgb_data[i*3:i*3+3]:
            pixel = FALSE
        else:
            pixel = TRUE
        bin_data.append(pixel)      
        return bin_data
        
## Handles a dropped Bitmap
def LoadBitmap():
    bitmap = wx.Image(filenames[0], wx.BITMAP_TYPE_BMP)
    rgb_data = bmp.GetData()
    

## File drag n' drop handler class
class FileDrop(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window
    def OnDropFiles(self, x, y, filenames):
        try:
            if len(filenames) > 1:
                raise MultipleFilesException()
            LoadBitmap()
        except MultipleFilesException:
            dlg = wx.MessageDialog(None, 'One file at a time please.')
            dlg.ShowModal()

class MainFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MainFrame.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.frame_1_statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        self.__set_properties()
        self.__do_layout()
        # end wxGlade

        drop_target = FileDrop(self)
        self.SetDropTarget(drop_target)
        
    def __set_properties(self):
        # begin wxGlade: MainFrame.__set_properties
        self.SetTitle("PyOCR")
        self.SetSize((473, 375))
        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.SetForegroundColour(wx.Colour(0, 0, 0))
        self.frame_1_statusbar.SetStatusWidths([-1])
        # statusbar fields
        frame_1_statusbar_fields = ["frame_1_statusbar"]
        for i in range(len(frame_1_statusbar_fields)):
            self.frame_1_statusbar.SetStatusText(frame_1_statusbar_fields[i], i)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: MainFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer_1)
        self.Layout()
        self.Centre()
        # end wxGlade

# end of class MainFrame


