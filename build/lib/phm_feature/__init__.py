from __future__ import absolute_import, unicode_literals
import scipy.fftpack
import scipy.signal
import scipy.stats
import logging
import sys
import threading
import numpy as np


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

    def __init__(self, name):
        self.lock = threading.RLock()
        self.name = name
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
        _kurtosis = scipy.stats.kurtosis(s, axis=-1, fisher=True, bias=True)
        _skewness = scipy.stats.skew(s, axis=-1, bias=True)
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
            yield array[:,:]
            start += hop_size
            end += hop_size
            if end > shape[1]:
                break

# default Tokenizer instance
dt = Tokenizer("ivystar")

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

def _pfeature_t(s):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(feature_t, _s)
    return result

def _pfeature_f(s, fs):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(feature_f, _s)
    return result

def _pfft(s, n):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(fft, (_s,n,))
    return result

def _ppower(s, n):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(power, _s)
    return result

def _pifft(s, n):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(ifft, _s)
    return result

def _pcepstrum(s, n):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(cepstrum, (_s, n,))
    return result

def _penvelope(s):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(envelope, _s)
    return result

def _pwindow(s, window_type):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(window, (_s, window_type,))
    return result

def _pdivide(s, window_size, hop_size):
    _s = s.reshape(-1, 1, s.shape[-1])
    result = pool.map(divide, (_s, window_size, hop_size,))
    return result

def enable_parallel(processnum=None):
    """
    Change the module's functions to the parallel version

    Note that this only works using dt, custom Tokenizer
    instances are not supported. 
    Auth: Qin Haining
    """
    global pool, feature_t, feature_f, fft, power, ifft, cepstrum, envelope, window, divide
    from multiprocessing import Pool
    pool = Pool(processnum)
    from multiprocessing import cpu_count
    if processnum is None:
        processnum = cpu_count()
    feature_t = _pfeature_t
    feature_f = _pfeature_f
    fft = _pfft
    power = _ppower
    ifft = _pifft
    cepstrum = _pcepstrum
    envelope = _penvelope
    window = _pwindow
    divide = _pdivide

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

