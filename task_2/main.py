import matplotlib.pyplot as plt
import numpy as np


def fourier_transform(lst):
    k = 0
    n = 0
    fourier_series = [0] * (len(lst))
    while k < len(lst):
        res = 0
        while n < len(lst):
            res += lst[n]*np.exp(2*np.pi*(-1)*k*n*1j/len(lst))
            n += 1
        fourier_series[k] = round(res.real, 2)+round(res.imag, 2) * 1j
        n = 0
        k += 1
    return fourier_series


def hertz(length, sampling):
    y = [0.0] * length
    td = 1 / sampling
    for i in range(len(y)):
        y[i] = i * td
    z = [0.0] * length
    for i in range(len(y)):
        z[i] = i / (td * len(y))
    return z


def convolution(f_lst, s_lst):
    f_lst_copy = np.concatenate((f_lst.copy(), np.zeros(len(s_lst) - 1)))
    s_lst_copy = np.concatenate((s_lst.copy(), np.zeros(len(f_lst) - 1)))
    c = [0.0] * (len(f_lst) + len(s_lst) - 1)
    for i in range(len(f_lst) + len(s_lst) - 1):
        for k in range(len(f_lst)):
                c[i] += f_lst_copy[k] * s_lst_copy[i - k]
    return c


def fourier_convolution(a, b):
    a_copy = fourier_transform(np.concatenate((a.copy(), np.zeros(len(b) - 1))))
    b_copy = fourier_transform(np.concatenate((b.copy(), np.zeros(len(a) - 1))))
    for i in range(len(a_copy)):
        a_copy[i] *= b_copy[i]
    a_copy = np.fft.ifft(a_copy)
    return a_copy


def fourier_correlation(a, b):
    a_copy = np.conj(fourier_transform(np.concatenate((a.copy(), np.zeros(len(b) - 1)))) - np.mean(a))
    b_copy = fourier_transform(np.concatenate((np.zeros(len(a) - 1), b.copy()))) - np.mean(b)
    for i in range(len(a_copy)):
        a_copy[i] *= b_copy[i]
    a_copy = np.fft.ifft(a_copy)
    return a_copy


def create_func_value_from_lst(*args, length, f):
    lst = [0.0] * length
    function_args = 1
    for i in args:
        function_args *= i
    for i in range(length):
        lst[i] = f(function_args * i)
    return lst


def create_meander(function, n, period, step):
    res = np.zeros(n)
    tmp = 0
    for i in range(n):
        if tmp < period:
            res[i] = function(tmp)
            tmp += step
        if tmp >= period:
            tmp = tmp % period
    return res


def main():

    w = 0.5
    td = 1/30
    length = 200
    z = hertz(2 * length - 1, 1/30)


    sin = create_func_value_from_lst(w * 2., np.pi, 2 * td, length=length, f=np.sin)
    cos = create_func_value_from_lst(w, np.pi, 2 * td, length=length, f=np.cos)

    T = 4
    A = 2
    meandr = create_meander(lambda x: A if x < T / 2 else -A, length, T, 1 / 4)
    meandr = meandr.tolist()
    meandr_2 = create_meander(lambda x: A if x < T / 2 else -A, length, T, 1 / 4)
    meandr_2 = meandr_2.tolist()
    meandr_3 = create_meander(lambda x: A if x < T / 2 else -A, 100, T, 1 / 4)
    meandr_3 = meandr_3.tolist()

    m1 = np.mean(meandr)
    m2 = np.mean(meandr_2)
    np_convolve = np.convolve(cos, sin)
    np_correlation = fourier_correlation(meandr[:193:], meandr_2[:193:])
    fourier_convolve = convolution(cos, sin + np.random.sample(len(cos)))
    fourier_correlate = fourier_correlation(meandr, meandr_2)
    np_correl = np.correlate(meandr[:192:], meandr_2[:192:], mode='full')
    user_convolve = convolution(cos, sin)
    user_convolve_meandr = fourier_convolution(meandr, meandr_3)
    user_convolve_meandr_2 = convolution(meandr, meandr_3)
    plt.subplot(141)
    plt.plot(np_correlation, '*')
    # plt.plot(fourier_convolve)

    plt.subplot(142)
    plt.plot(meandr[:192:])
    plt.plot(meandr_2[:192:])
    # plt.plot(z, fourier_convolve)

    plt.subplot(143)
    plt.plot(fourier_convolve)

    plt.subplot(144)
    plt.plot(fourier_convolve - np.mean(fourier_convolve))


    plt.show()


if __name__ == '__main__':
    main()
