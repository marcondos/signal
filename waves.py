import numpy as np

class Wave(object):
    pass

class AnalogWave(Wave):
    def __init__(self, freq, phase=0):
        self.phase = phase
        self.frequency = freq
        self.angular_frequency = 2 * np.pi * freq

    def plot(self, length, ax, **kwargs):
        continuous_time = np.linspace(0, length, 300)
        ax.plot(continuous_time, self.function(continuous_time), **kwargs)

class AnalogSineWave(AnalogWave):
    Amin = -1
    Amax = 1
    amplitude = Amax - Amin
    def function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class ADC(object):
    pass

class PCMSampler(ADC):
    def __init__(self, freq, res):
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq
        self.amplitude_resolution = res
        self.levels = 2**res

    def sample(self, signal, time, ax, **kwargs):
        # signal is a AnalogWave object
        time_domain = np.arange(0, time, self.sampling_interval)
        sampled_signal = signal.function(time_domain)
        sampled_signal = finite_resolution(sampled_signal, self.levels, signal)
        t = time_domain.repeat(2)[1:]
        y = sampled_signal.repeat(2)[:-1]
        ax.plot(t, y, **kwargs)

class CDFormatSampler(PCMSampler):
    def __init__(self):
        super().__init__(44100, 16)

def finite_resolution(sampled, levels, signal):
    A = signal.amplitude
    int_ampl = levels - 1
    unsigned = np.array([round((s - signal.Amin) * int_ampl/A) for s in sampled])
    signed = unsigned * A/int_ampl + signal.Amin
    return signed
