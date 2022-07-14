#!encoding=utf-8
import phm_feature
from phm_feature import enable_parallel,disable_parallel
import numpy as np
from phm_feature import *
import unittest

class TestSetUp(unittest.TestCase):
    '''
    每一个测试用例方法执行之前都会运行setUp()
    可以把测试数据放到 setUp 当中
    setUp() ==> test_xxx() ==> tearDown()
    '''

    def setUp(self):
        pass

    def test_parallel_feature(self):
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        enable_parallel(processnum=None)
        feature_t(data)
        #feature_f
        fft(data, 50)
        power(data, 50)
        ifft(data, 50)
        cepstrum(data, 50)
        envelope(data)
        window(data, 'hamming')
        divide(data, 50, 25)

    def test_disable_parallel_feature(self):
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        enable_parallel(processnum=None)
        disable_parallel()
        feature_t(data)
        #feature_f
        fft(data, 50)
        power(data, 50)
        ifft(data, 50)
        cepstrum(data, 50)
        envelope(data)
        window(data, 'hamming')
        divide(data, 50, 25)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

