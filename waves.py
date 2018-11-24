import numpy as np

class Wave(object):
    def __init__(self, label):
        self.label = label

class AnalogWave(Wave):
    def __init__(self, label, freq, phase=0):
        self.phase = phase
        self.frequency = freq
        self.angular_frequency = 2 * np.pi * freq
        super().__init__(label)

    def plot(self, length, ax, **kwargs):
        continuous_time = np.linspace(0, length, 300)
        ax.plot(continuous_time, self.function(continuous_time),
                label=self.label, **kwargs)

class AnalogSineWave(AnalogWave):
    Amin = -1
    Amax = 1
    amplitude = Amax - Amin
    def function(self, t):
        return np.sin(self.angular_frequency * t + self.phase)

class ADC(object):
    def __init__(self, label, freq, res):
        # frequency in Hz, resolution in bits
        self.sampling_frequency = freq
        self.sampling_interval = 1/freq
        self.amplitude_resolution = res
        self.levels = 2**res
        self.label = label

class DSDSampler(ADC):
    pass
    # noise shaping

class SACDFormatSampler(DSDSampler):
    def __init__(self, label)
        super().__init__(label, 64 * 44100, 1):

class PCMSampler(ADC):
    def sample(self, signal, time, ax, **kwargs):
        # signal is a AnalogWave object
        time_domain = np.arange(0, time, self.sampling_interval)
        sampled_signal = signal.function(time_domain)
        sampled_signal, stream = finite_resolution(sampled_signal, self.levels,
                signal, self.label)
        t = time_domain.repeat(2)[1:]
        y = sampled_signal.repeat(2)[:-1]
        ax.plot(t, y, **kwargs, label=self.label)
        return stream

class CDFormatSampler(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 44100, 16)

class HFPALinearPCM(PCMSampler):
    def __init__(self, label):
        super().__init__(label, 192000, 24)

def finite_resolution(sampled, levels, signal, label):
    A = signal.amplitude
    int_ampl = levels - 1
    unsigned = np.array([round(s * int_ampl/A) for s in sampled-signal.Amin],
            dtype=int)
    print(label, 'stream:', unsigned)
    signed = unsigned * A/int_ampl + signal.Amin
    return signed, unsigned
