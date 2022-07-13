#!
import numpy as np
import scipy.stats

# build data
sin_arr = np.sin(np.linspace(0,360,100)*np.pi/360)
sin_arr_mat = np.repeat(sin_arr, 10).reshape(10,100)

# mean,max,min,std,median,p2p
_mean = np.mean(sin_arr_mat, axis=-1)
_max = np.max(sin_arr_mat, axis=-1)
_min = np.min(sin_arr_mat, axis=-1)
_std = np.std(sin_arr_mat, axis=-1)
_median = np.median(sin_arr_mat, axis=-1)
_p2p = _max - _min
_rms = np.mean(sin_arr_mat**2, axis=-1)
_abs_max = np.abs(np.max(sin_arr_mat, axis=-1))
_abs_min = np.abs(np.min(sin_arr_mat, axis=-1))
_x_p = np.max(np.vstack([_abs_max, _abs_min]), axis=0)
_arv = np.mean(np.abs(sin_arr_mat), axis=-1)
_r = np.mean(np.sqrt(np.abs(sin_arr_mat)), axis=-1) ** 2

_kurtosis = scipy.stats.kurtosis(sin_arr_mat, axis=-1, fisher=True, bias=True)
_skewness = scipy.stats.skew(sin_arr_mat, axis=-1, bias=True)
_pulse_factor = _x_p/_arv
_margin_factor = _x_p/_r
_form_factor = _rms/_arv






