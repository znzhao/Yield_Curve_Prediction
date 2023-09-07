import os
import time
import numpy as np
import datetime

maturities = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

class Timer(object):
    def __init__(self, name=None, display = True):
        self.name = name
        self.display = display

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name and self.display:
            print('[{}] '.format(self.name), end='\r')
        if self.display: print('Elapsed in {:.4f} secs.'.format(time.time() - self.tstart))


def getYahooTicker(name):
    yahoo_ticker = {
        'SP500': '^GSPC'
    }
    if name in yahoo_ticker.keys():
        return yahoo_ticker[name]
    else:
        return name

def dropCommonNan(true_y: np.array, pred_y: np.array):
    true_y = true_y.flatten()
    pred_y = pred_y.flatten()
    nanindex_true = set(np.argwhere(np.isnan(true_y)).flatten().tolist())
    nanindex_pred = set(np.argwhere(np.isnan(pred_y)).flatten().tolist())
    nanindex = list(nanindex_true.union(nanindex_pred))
    true_y = np.delete(true_y, nanindex)
    pred_y = np.delete(pred_y, nanindex)
    return true_y, pred_y

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

def interpolateColor(color1, color2, alpha):
    color1 = hex_to_rgb(color1)
    color2 = hex_to_rgb(color2)
    newcolor = tuple(np.array(color1)*(1-alpha) + np.array(color2)*alpha)
    return rgb_to_hex(int(newcolor[0]), int(newcolor[1]), int(newcolor[2]))

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            fullname = fullname.replace("\\","/")
            yield fullname

def nextBusinessDay(date):
    if date.weekday() == 4:
        date = date + datetime.timedelta(days=3)
    else:
        date = date + datetime.timedelta(days=1)
    return date

def nextKBusinessDay(date, k):
    if k == 1:
        return nextBusinessDay(date)
    else:
        return nextKBusinessDay(nextBusinessDay(date), k-1)

if __name__ == "__main__":
    print('Data Tools loaded.')