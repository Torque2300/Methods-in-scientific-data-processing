import matplotlib.pyplot
import numpy


def fourier_transform(lst):
    k = 0
    n = 0
    fourier_series = [0] * (len(lst))
    while k < len(lst):
        res = 0
        while n < len(lst):
            res += lst[n]*numpy.exp(2*numpy.pi*(-1)*k*n*1j/len(lst))
            n += 1
        fourier_series[k] = round(res.real, 2)+round(res.imag, 2) * 1j
        n = 0
        k += 1
    return fourier_series


def main():
    lst = [0, numpy.pi]

    user_fourier = fourier_transform(numpy.cos(lst))
    numpy_fourier = numpy.fft.fft(numpy.cos(lst))

    for i in range(len(user_fourier)):
        assert numpy_fourier.flat[i] == user_fourier[i]

    #plot of a transformation
    y = [0.0] * 200
    td = 1/30
    fd = 1 / td
    w = 1
    for i in range(len(y)):
        y[i] = i * td

    four_list = [0.0] * 200

    rand_arr = numpy.random.sample(200)

    f_func = [0.0] * 200
    s_func = [0.0] * 200
    for i in range(200):
        four_list[i] = 3 * numpy.sin(2*numpy.pi*w*i*td) + 9 * numpy.sin(2*numpy.pi*10*i*td)
        f_func[i] = numpy.sin(2*numpy.pi*w*i*td)
        s_func[i] = numpy.cos(2*numpy.pi*(12+w)*i*td)

    four_list_copy = four_list.copy()
    four_list = fourier_transform(four_list+rand_arr-numpy.mean(rand_arr))
    four_without_noise = fourier_transform(four_list_copy + rand_arr)
    ampl_four_list = [0.0] * 200
    for i in range(200):
        ampl_four_list[i] = abs(four_list[i])


    z = [0.0] * 200
    for i in range(len(y)):
        z[i] = i / (td * len(y))
    matplotlib.pyplot.subplot(131)
    matplotlib.pyplot.plot(z, ampl_four_list)
    matplotlib.pyplot.xlabel('Hertz, Hz')
    matplotlib.pyplot.ylabel('Amplitude')
    matplotlib.pyplot.subplot(132)
    matplotlib.pyplot.plot(z, numpy.abs(four_without_noise))
    matplotlib.pyplot.xlabel('Hertz, Hz')
    matplotlib.pyplot.ylabel('Amplitude')
    matplotlib.pyplot.subplot(133)
    matplotlib.pyplot.plot(z, numpy.abs(four_without_noise))
    matplotlib.pyplot.plot(z, ampl_four_list)
    matplotlib.pyplot.xlabel('Hertz, Hz')
    matplotlib.pyplot.ylabel('Amplitude')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()
