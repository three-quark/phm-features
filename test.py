#!encoding=utf-8
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

    def test_disable_parallel_feature(self):
        print(">>> test_disable_parallel_feature")
        from phm_feature import enable_parallel, disable_parallel
        enable_parallel(processnum=None)
        disable_parallel()
        from phm_feature import yin,feature_t,fft,power,ifft,cepstrum,envelope,window,divide
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        feature_t(data)
        _ = yin(data, window_size=25)
        print(_.shape)
        fft(data, num=100)
        power(data, 50)
        ifft(data, 50)
        cepstrum(data, 50)
        envelope(data)
        window(data, 'hamming')
        _ = divide(data, 50, 25)

    def test_parallel_feature(self):
        print(">>> test_enable_parallel_feature")
        from phm_feature import enable_parallel
        enable_parallel(processnum=None)
        from phm_feature import yin,feature_t,fft,power,ifft,cepstrum,envelope,window,divide
        data = np.sin(np.linspace(0,360,300)/360*np.pi*2).reshape(3,100)
        feature_t(data)
        _ = yin(data, window_size=25)
        print(_.shape)
        fft(data, num=100)
        power(data, 50)
        ifft(data, 50)
        cepstrum(data, 50)
        envelope(data)
        window(data, 'hamming')
        _ = divide(data, 50, 25)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

