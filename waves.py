import numpy as np
import itertools
from scipy import interpolate

COLOR = itertools.cycle(['C%i' % i for i in range(10)])

class Wave(object):
    def __init__(self, label, gain=1):
        self.label = label
        self.gain = gain

    def function(self, *args, **kwargs):
        return self.gain * self._function(*args, **kwargs)

class AnalogWave(Wave):

    infinite_points = 400

    def __init__(self, label, freq, gain=1, phase=0):
        self.phase = phase
        self.frequency = freq
        self.angular_frequency = 2 * np.pi * freq
        super().__init__(label, gain=gain)

    def time_array(self, length):
        return np.linspace(0, length, self.infinite_points)

    def plot(self, length, ax, **kwargs):
        continuous_time = self.time_array(length)
        ax.plot(continuous_time, self.function(continuous_time),
                label=self.label, **kwargs)
        return continuous_time

class AnalogSineWave(AnalogWave):
    def _function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class ADC(object):
    def __init__(self, label, freq, res, amp_max=1.2, color=None):
        self.color = next(COLOR) if color is None else color
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq
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

    def quantize(self, input_signal, time_length, dither=0):
        time_domain = self.discrete_time(time_length)
        if dither > 0:
            dither = np.random.triangular(-dither, 0, dither,
                    size=time_domain.size)

        # classification stage:
        unsigned = np.array([round(s/self.quantization_step) for s in self.clip(
            input_signal.function(time_domain) + dithering
            ) - self.amp_min], dtype=int)
        print(of(self.label), 'stream:', shortlist(unsigned))

        # reconstruction stage:
        signed = unsigned * self.quantization_step + self.amp_min
        time = input_signal.time_array(time_length)
        analog = input_signal.function(time)
        digital = interpolate.interp1d(time_domain, signed, kind=0,
                fill_value='extrapolate')
        quantization_noise = digital(time) - analog
        return time_domain, signed, unsigned, quantization_noise

class DSDSampler(ADC):
    pass
    # noise shaping

class SACDFormatSampler(DSDSampler):
    def __init__(self, label):
        super().__init__(label, 64 * 44100, 1)

class PCMSampler(ADC):
    def discrete_time(self, length):
        return np.arange(0, length, self.sampling_interval)

    def sample(self, signal, time_length, ax, dither=0, **kwargs):
        # signal is a AnalogWave object
        sampled_time, sampled_signal, stream, noise = self.quantize(signal,
                time_length, dither=dither)
        t = sampled_time.repeat(2)[1:]
        y = sampled_signal.repeat(2)[:-1]
        kwargs.pop('color', None)
        ax.plot(t, y, **kwargs, color=self.color, label=self.label)
        for kw in ['lw', 'linewidth', 'ls', 'linestyle']:
            kwargs.pop(kw, None)
        signal_domain = signal.time_array(time_length)
        ax.plot(signal_domain, noise, lw=0.5, ls='-', color=self.color, 
                label=of(self.label) + ' noise', **kwargs)
        return stream, sampled_time, signal_domain, noise

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

def shortlist(array):
    return ' '.join([str(a) for a in list(array[:5]) + ['...'] + list(array[-5:])]).join(['[', ']'])
