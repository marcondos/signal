import numpy as np
import itertools
from scipy import interpolate

COLOR = itertools.cycle(['C%i' % i for i in range(10)])

class Wave(object):
    def __init__(self, label):
        self.label = label

class AnalogWave(Wave):

    infinite_points = 400

    def __init__(self, label, freq, phase=0):
        self.phase = phase
        self.frequency = freq
        self.angular_frequency = 2 * np.pi * freq
        super().__init__(label)

    def time_array(self, length):
        return np.linspace(0, length, self.infinite_points)

    def plot(self, length, ax, **kwargs):
        continuous_time = self.time_array(length)
        ax.plot(continuous_time, self.function(continuous_time),
                label=self.label, **kwargs)

class AnalogSineWave(AnalogWave):
    Amin = -1
    Amax = 1
    amplitude = Amax - Amin
    def function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class ADC(object):
    def __init__(self, label, freq, res, color=None):
        self.color = next(COLOR) if color is None else color
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq
        self.amplitude_resolution = res
        self.levels = 2**res
        self.int_amplitude = self.levels - 1
        self.label = label

    def quantize(self, input_signal, time_length):
        time_domain = self.discrete_time(time_length)

        # classification stage:
        unsigned = np.array([round(s \
                * self.int_amplitude/input_signal.amplitude) \
                for s in input_signal.function(time_domain) - input_signal.Amin],
                dtype=int)
        print(of(self.label), 'stream:', unsigned)

        # reconstruction stage:
        signed = unsigned * input_signal.amplitude/self.int_amplitude \
                + input_signal.Amin
        time = input_signal.time_array(time_length)
        analog = input_signal.function(time)
        digital = interpolate.interp1d(time_domain, signed, kind=0,
                fill_value='extrapolate')
        quantization_noise = digital(time) - analog
        return time_domain, signed, unsigned, quantization_noise





        # Dither

class DSDSampler(ADC):
    pass
    # noise shaping

class SACDFormatSampler(DSDSampler):
    def __init__(self, label):
        super().__init__(label, 64 * 44100, 1)

class PCMSampler(ADC):
    def discrete_time(self, length):
        return np.arange(0, length, self.sampling_interval)

    def sample(self, signal, time_length, ax, **kwargs):
        # signal is a AnalogWave object
        sampled_time, sampled_signal, stream, noise = self.quantize(signal,
                time_length)
        t = sampled_time.repeat(2)[1:]
        y = sampled_signal.repeat(2)[:-1]
        kwargs.pop('color', None)
        ax.plot(t, y, **kwargs, color=self.color, label=self.label)
        for kw in ['lw', 'linewidth', 'ls', 'linestyle']:
            kwargs.pop(kw, None)
        ax.plot(signal.time_array(time_length), noise, lw=0.5, ls='-',
                color=self.color, 
                label=of(self.label) + ' noise', **kwargs)
        return stream, sampled_time, noise

class CDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 16)

class HDCDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 20)

class HFPALinearPCM(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 192000, 24)

def quantize(sampled, levels, signal, length, discrete_time, label):
    A = signal.amplitude
    int_ampl = levels - 1
    # classification stage:
    unsigned = np.array([round(s * int_ampl/A) for s in sampled-signal.Amin],
            dtype=int)
    print(label, 'stream:', unsigned)
    # reconstruction stage:
    signed = unsigned * A/int_ampl + signal.Amin
    time = signal.time_array(length)
    analog = signal.function(time)
    digital = interpolate.interp1d(discrete_time, signed, kind=0,
            fill_value='extrapolate')
    quant_noise = digital(time) - analog
    return signed, unsigned, quant_noise

def of(name):
    return name + ("'" if name.endswith('s') else "'s")
