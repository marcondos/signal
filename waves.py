import numpy as np
import itertools
from scipy import interpolate
from math import floor

COLOR = itertools.cycle(['C%i' % i for i in range(10)])

class Wave(object):
    def __init__(self, label, gain=1):
        self.label = label
        self.gain = gain

    def function(self, *args, **kwargs):
        return self.gain * self._function(*args, **kwargs)

class AnalogWave(Wave):

    def __init__(self, label, freq, gain=1, phase=0, infinity=400):
        self.phase = phase
        self.frequency = freq
        self.period = 1/self.frequency
        self.angular_frequency = 2 * np.pi * freq
        self.infinity = infinity
        super().__init__(label, gain=gain)

    def time_array(self, length):
        return np.arange(0, length, self.period/self.infinity)

    def plot(self, length, ax, **kwargs):
        continuous_time = self.time_array(length)
        ax.plot(continuous_time*1000, self.function(continuous_time),
                label=self.label, **kwargs) # in plot, time in miliseconds
        return continuous_time

class AnalogSineWave(AnalogWave):
    def _function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class DigitalWave(Wave):

    def __init__(self, label, time, signal, stream):
        self.label = label
        self.stream = stream
        self.sampled_time = time
        self.sampled_signal = signal

class ADC(object):
    def __init__(self, label, freq, res, amp_max=1.2, color=None):
        self.color = next(COLOR) if color is None else color
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
        return DigitalWave(input_signal.label + ' converted', time_domain,
                signed, unsigned), quantization_noise

class DSDSampler(ADC):
    pass
    # noise shaping

class SACDFormatSampler(DSDSampler):
    def __init__(self, label):
        super().__init__(label, 64 * 44100, 1)

class PCMSampler(ADC):
    def discrete_time(self, length):
        return np.arange(0, length, self.sampling_interval)

    def sample(self, signal, time_length, ax, show_noise=False, dither=False,
            **kwargs):
        # signal is a AnalogWave object
        converted, noise = self.quantize(signal,
                time_length, dither=dither)
        t = converted.sampled_time.repeat(2)[1:]
        y = converted.sampled_signal.repeat(2)[:-1]
        kwargs.pop('color', None)
        ax.plot(t*1000, y, **kwargs, color=self.color, label=self.label) #t (ms)
        for kw in ['lw', 'linewidth', 'ls', 'linestyle']:
            kwargs.pop(kw, None)
        signal_domain = signal.time_array(time_length)
        if show_noise:
            ax.plot(signal_domain*1000, noise, lw=0.5, ls='-', color=self.color,
                    label=of(self.label) + ' noise', **kwargs)
        return converted, noise

class CDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 16)

class HDCDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 20)

class HFPALinearPCM(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 192000, 24)

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
