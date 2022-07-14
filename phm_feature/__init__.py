from __future__ import absolute_import, unicode_literals
import scipy.fftpack
import scipy.signal
import scipy.stats
import logging
import sys
import threading
import numpy as np
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import os


__version__ = '0.0.1'
__license__ = 'MIT'
__name__ = 'phm_feature'


_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

def setLogLevel(log_level):
    default_logger.setLevel(log_level)

class Tokenizer(object):

    def __init__(self, name=__name__):
        # self.lock = threading.RLock()
        self.name = __name__
        self.time_feature_name = ["mean","max","min","std","median","p2p","rms","x_p","arv","r","kurtosis","skewness","pulse_factor","margin_factor","form_factor"]

    def __repr__(self):
        return 'Tokenizer of vibration-wave {}'.format(self.name)

    def feature_t(self, s):
        _mean = np.mean(s, axis=-1)
        _max = np.max(s, axis=-1)
        _min = np.min(s, axis=-1)
        _std = np.std(s, axis=-1)
        _median = np.median(s, axis=-1)
        _p2p = _max - _min
        _rms = np.mean(s**2, axis=-1)
        _abs_max = np.abs(np.max(s, axis=-1))
        _abs_min = np.abs(np.min(s, axis=-1))
        _x_p = np.max(np.vstack([_abs_max, _abs_min]), axis=0)
        _arv = np.mean(np.abs(s), axis=-1)
        _r = np.mean(np.sqrt(np.abs(s)), axis=-1) ** 2
        _kurtosis = 1 #scipy.stats.kurtosis(s, axis=-1, fisher=True, bias=True)
        _skewness = 1 #scipy.stats.skew(s, axis=-1, bias=True)
        _pulse_factor = _x_p/_arv
        _margin_factor = _x_p/_r
        _form_factor = _rms/_arv
        return _mean, _max, _min, _std, _median, _p2p, _rms, _x_p, _arv, _r, _kurtosis, _skewness, _pulse_factor, _margin_factor, _form_factor

    def feature_f(self, array, fs):
        raise Exception('oh friend, sorry, this functions is not finish yet')

    def fft(self, array, num):
        return scipy.fftpack.fft(array, num)

    def power(self, array, num):
        return (np.abs(scipy.fftpack.fft(array, num))**2)/num

    def ifft(self, array, num):
        return scipy.fftpack.ifft(array, num)

    def cepstrum(self, array, num):
        '''cepstrum
        signal->power->log->ifft
        '''
        spectrum = scipy.fftpack.fft(array, num)
        ceps = scipy.fftpack.ifft(np.log(np.abs(spectrum))).real
        return ceps

    def envelope(self, s):
        xh = scipy.signal.hilbert(s)
        xe = np.abs(xh)
        xe = xe - np.mean(xe, axis=0)
        xh3 = np.fft.rfft(xe) / len(xe)
        mag = np.abs(xh3) * 2
        #fre = np.linspace(0, fs / 2, int(len(xe) / 2 + 1))
        return mag

    def yin(self, s, window_size):
        yin_s = []
        assert window_size < s.shape[-1]
        start_l, end_l = 0, window_size
        start_r, end_r = 0, window_size
        while(True):
            yin_s.append(np.subtract(s[:,end_l:end_r], s[:,start_l:start_r]))
            end_r+=1
            end_l+=1
            if end_r > s.shape[-1]:
                break
        return np.array(yin_s).T

    def fs2Fs(self, L, fs):
        ''' convert fs 2 Freqs list '''
        fre = np.linspace(0, fs / 2, int(L / 2 + 1))
        return fre

    def window(self, array, window_type='hamming'):
        if window_type == "hamming":
            return np.multiply(np.hamming(array.shape[-1]), array)
        else:
            raise Exception('Oh, my friend. u should specified the window name first')

    def divide(self, array, window_size, hop_size):
        ''' request hop_size is (0,0.5)*window_size
        '''
        start,end,shape = 0,window_size,array.shape
        while(True):
            yield array[:,start:end]
            start += hop_size
            end += hop_size
            if end > shape[1]:
                break

# default Tokenizer instance
dt = Tokenizer()

# global functions
feature_t = dt.feature_t
feature_f = dt.feature_f
fft = dt.fft
power = dt.power
ifft = dt.ifft
cepstrum = dt.cepstrum
envelope = dt.envelope
window = dt.window
divide = dt.divide
yin = dt.yin

def _feature_t(s):
    return dt.feature_t(s)

def _pyin(s, window_size):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(yin, window_size=window_size), _s)
    return result

def _pfeature_t(s):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(_feature_t, _s)
    return result

def _pfeature_f(s, fs):
    pass

def _pfft(s, num):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.fft, num=num), _s)
    return result

def _ppower(s, n):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.power, num=n), _s)
    return result

def _pifft(s, n):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.ifft, num=n), _s)
    return result

def _pcepstrum(s, n):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.cepstrum, num=n), _s)
    return result

def _penvelope(s):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(dt.envelope, _s)
    return result

def _pwindow(s, window_type):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.window, window_type=window_type), _s)
    return result

def _pdivide(s, window_size, hop_size):
    _s = [np.array(_).reshape(1,-1) for _ in s.tolist()]
    result = pool.map(partial(dt.divide, window_size=window_size, hop_size=hop_size), _s)
    return result

def enable_parallel(processnum=None):
    """
    Change the module's functions to the parallel version

    Note that this only works using dt, custom Tokenizer
    instances are not supported. 
    Auth: Qin Haining
    """
    global pool, feature_t, feature_f, fft, power, ifft, cepstrum, envelope, window, divide
    if os.name == 'nt':
        raise NotImplementedError(
            "parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    #print(_pfeature_t)
    feature_t = _pfeature_t
    #feature_f = _pfeature_f
    fft = _pfft
    power = _ppower
    ifft = _pifft
    cepstrum = _pcepstrum
    envelope = _penvelope
    window = _pwindow
    divide = _pdivide
    yin = _pyin
    print(">>> enable_parallel")

def disable_parallel():
    global pool, feature_t, feature_f, fft, power, ifft, cepstrum, envelope, window, divide
    if pool:
        pool.close()
        pool = None
    feature_t = dt.feature_t
    feature_f = dt.feature_f
    fft = dt.fft
    power = dt.power
    ifft = dt.ifft
    cepstrum = dt.cepstrum
    envelope = dt.envelope
    window = dt.window
    divide = dt.divide
    yin = dt.yin
    print(">>> disable_parallel")

