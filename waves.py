import numpy as np

class Wave(object):
    pass

class AnalogWave(Wave):
    def __init__(self, freq, phase=0):
        self.phase = phase
        self.frequency = freq
        self.angular_frequency = 2 * np.pi * freq

    def plot(self, ax, time, signal, **kwargs):
        ax.plot(time, signal, **kwargs)

class AnalogSineWave(AnalogWave):
    def function(t):
        return np.sin(self.angular_frequency * t + self.phase)

class Sampler(object):
    pass

class PCMSampler(Sampler):
    def __init__(self, freq, res):
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq
        self.amplitude_resolution = res
        self.levels = 2**res

    def plot(self, ax, time, signal, **kwargs):
        # signal is a AnalogWave object
        continuous_time = np.linspace(0, time, 300)
        ax.plot(continuous_time, signal.function(continuous_time), color='k',
                lw=1)
        time_domain = np.arange(0, time, self.sampling_interval)
        sampled_signal = signal.function(time_domain)
        t = time_domain.repeat(2)[1:]
        y = sampled_signal.repeat(2)[:-1]
        ax.plot(t, y, **kwargs)

class CDFormatSampler(PCMSampler):
    def __init__(self):
        super().__init__(44100, 16)
