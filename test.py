#!encoding=utf-8
from phm_feature import enable_parallel
enable_parallel(processnum=None)

from phm_feature import feature_t
from phm_feature import fft
import numpy as np
import unittest

class TestSetUp(unittest.TestCase):
    '''
    每一个测试用例方法执行之前都会运行setUp()
    可以把测试数据放到 setUp 当中
    setUp() ==> test_xxx() ==> tearDown()
    '''

    def setUp(self):
        pass

    def _test_yin(self):
        data = np.random.randint(0,100,100).reshape(2,50)
        res = yin(data, 10)
        print(res)

    def test_parallel_feature(self):
        print(">>> test_parallel_feature")
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        print(feature_t)
        feature_t(data)
        fft(data, num=100)
        #print(power)
        #print(ifft)
        #print(cepstrum)
        #feature_f
        #fft(data, 50)
        #power(data, 50)
        #ifft(data, 50)
        #cepstrum(data, 50)
        #envelope(data)
        #window(data, 'hamming')
        #divide(data, 50, 25)

    def _test_disable_parallel_feature(self):
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        enable_parallel(processnum=None)
        print(enable_parallel(processnum=None))
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

