import numpy as np
from scipy import interpolate
from math import floor

class Wave(object):
    def __init__(self, label):
        self.label = label

class AnalogWave(Wave):

    def __init__(self, label, freq, gain=1, phase=0, infinity=400):
        self.gain = gain
        self.phase = phase
        self.frequency = freq
        self.period = 1/self.frequency
        self.angular_frequency = 2 * np.pi * freq
        self.infinity = infinity
        super().__init__(label)

    def function(self, *args, **kwargs):
        return self.gain * self._function(*args, **kwargs)

    def time_array(self, length):
        return np.arange(0, length, self.period/self.infinity)

    def view(self, ax, length, **kwargs):
        continuous_time = self.time_array(length)
        label = kwargs.pop('label', self.label)
        ax.plot(continuous_time*1000, self.function(continuous_time),
                label=label, **kwargs) # in plot, time in miliseconds
        return continuous_time

class AnalogSineWave(AnalogWave):
    def _function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class DigitalWave(Wave):
    
    def view(self, ax, **kwargs):
        t = self.t.repeat(2)[1:]
        y = self.y.repeat(2)[:-1]
        ax.plot(t*1000, y, **kwargs) # t (ms)

class PCMDigitalWave(DigitalWave):

    def __init__(self, label, stream, rate, bits):
        self.stream = stream
        self.sampling_rate = rate
        self.sampling_resolution = bits
        self.levels = 2**bits
        super().__init__(label)

class ADC(object):
    def __init__(self, label, freq, res, amp_max=1.2):
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq # in seconds
        self.amplitude_resolution = res
        self.levels = 2**res
        int_amplitude = self.levels - 1
        self.amp_max = amp_max
        self.amp_min = -amp_max
        amplitude = self.amp_max - self.amp_min
        self.quantization_step = amplitude/int_amplitude
        self.label = label

    def clip(self, signal):
        return np.clip(signal, -self.amp_max, self.amp_max)

    def quantize(self, input_signal, time_length, dither=False):
        time_domain = self.discrete_time(time_length)
        # lowest power ideal dither, d = 1/step, so the TPDF amplitude is
        # twice the quantization step

        # Triangular or retangular distributions require lower level of added
        # noise for full elimination of distortion than Gaussian noise;
        # Triangular distributed noise also minimizes noise modulation
        # retangular:
        distribution, pars = np.random.uniform, (0, self.quantization_step)
        # triangular:
        #distribution, pars = np.random.triangular, (0, self.quantization_step/2, self.quantization_step)
        dithering = distribution(*pars, size=time_domain.size) \
                if dither else 0

        # classification stage:
        clipped_signal = self.clip(input_signal.function(time_domain) \
                + dithering) - self.amp_min
        unsigned = np.array([floor(s/self.quantization_step) \
                for s in clipped_signal], dtype=int)
        #print(of(self.label), 'stream (%i levels):' % self.levels,
        #        shortlist(unsigned))

        # reconstruction stage:
        signed = unsigned * self.quantization_step + self.amp_min
        time = input_signal.time_array(time_length)
        analog = input_signal.function(time)
        digital = interpolate.interp1d(time_domain, signed, kind=0,
                fill_value='extrapolate')
        quantization_noise = digital(time) - analog
        digital = self.digitize(input_signal.label + ' converted', unsigned,
                self.sampling_frequency, self.amplitude_resolution)
        digital.t = time_domain
        digital.y = signed
        # *pars
        return digital, quantization_noise

class DSDSampler(ADC):
    pass
    # noise shaping

class SACDFormatSampler(DSDSampler):
    def __init__(self, label):
        super().__init__(label, 64 * 44100, 1)

class PCMSampler(ADC):
    digitize = PCMDigitalWave

    def discrete_time(self, length):
        return np.arange(0, length, self.sampling_interval)

class CDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 16)

class HDCDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 20)

class HFPALinearPCM(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 192000, 24)

class DAC(object):
    pass

class DeltaSigma(DAC):
    pass

class OneBitDeltaSigma(DeltaSigma):
    pass

class MultiBitDeltaSigma(DeltaSigma):
    pass

class R2R(DAC):
    pass

def of(name):
    return name + ("'" if name.endswith('s') else "'s")

def correlation(X, Y, j):
    Xbar, Ybar = X.mean(), Y.mean()
    SX, SY = X.size, Y.size
    assert SX == SY
    rho = np.sum( (X[:SX-j]-Xbar) * (Y[j:]-Ybar) )
    rho /= np.sqrt( np.sum( (X[:SX-j]-Xbar)**2 ) )
    rho /= np.sqrt( np.sum( (Y[j:]-Ybar)**2 ) )
    return rho

def twopar_correlation(X, Y):
    lagsize = X.size
    assert Y.size == lagsize
    return list(range(lagsize//2)), [correlation(X, Y, lag) for lag in range(lagsize//2)]

def shortlist(array, show=4):
    return ' '.join([str(a) for a in list(
        array[:show]) + ['...'] + list(array[-show:])]).join(['[', ']'])
