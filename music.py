from animation import *
import numpy as np
from scipy.io.wavfile import read

class Music():
    def __init__(self, input_file='input.wav', framerate=15):
        sample_rate, wave = read(input_file)
        samples_per_frame = int(sample_rate / framerate)
        window_size = 2 ** (np.floor(np.log2(sample_rate)) - 1)

        pad_width = sample_rate # 1 second of 0 padding
        wave = [0] * pad_width + list(wave) + [0] * pad_width
        wave = np.array(wave)
        centers = np.arange(pad_width, len(wave) - pad_width, samples_per_frame)

        from tqdm import tqdm
        ffts = [np.real(np.fft.rfft(wave[center-window_size:center+window_size])) for center in tqdm(centers)]
        ffts = np.stack(ffts)
        ffts = np.log(ffts)
        ffts[np.isnan(ffts)] = 0
        ffts[np.isinf(ffts)] = 0
        ffts[np.isneginf(ffts)] = 0

        bins = np.cast['int'](np.logspace(0, np.log(ffts.shape[1]), 16+1, endpoint=True, base=np.e))
        edges = [(int(bins[i]), int(bins[i+1])) for i in range(len(bins) - 1)]
        # sqrt of the mean exp, to give high values more impact, but not outrageously much
        binned = [[np.sqrt(np.mean(np.exp(row[a:b]))/10) for a, b in edges] for row in tqdm(ffts)]
        binned = np.array(binned)
        #binned[np.isnan(binned)] = 0

        self.smoothed_all = np.array([binned[max(0, i-3):i+3, :].mean(axis=0) for i in range(len(binned))])

        self.bass = binned[:,5:8].mean(axis=1)
        self.mid = binned[:,8:10].mean(axis=1)
        self.high = binned[:,10:12].mean(axis=1)
        self.freqs = np.stack([self.bass, self.mid, self.high])

        self.smoothed_bass = np.array([self.bass[max(0, i-3):i+3].mean() for i in range(len(self.bass))])
        self.smoothed_mid = np.array([self.mid[max(0, i-3):i+3].mean() for i in range(len(self.mid))])
        self.smoothed_high = np.array([self.high[max(0, i-3):i+3].mean() for i in range(len(self.high))])
        self.smoothed = np.stack([self.smoothed_bass, self.smoothed_mid, self.smoothed_high])

        self.bass_dev = binned[:,4:9].std(axis=1)
        self.mid_dev = binned[:,7:11].std(axis=1)
        self.high_dev = binned[:,9:13].std(axis=1)
        self.devs = np.stack([self.bass_dev, self.mid_dev, self.high_dev])

        self.smoothed_bass_dev = np.array([self.bass_dev[max(0, i-2):i+2].mean() for i in range(len(self.bass_dev))])
        self.smoothed_mid_dev = np.array([self.mid_dev[max(0, i-2):i+2].mean() for i in range(len(self.mid_dev))])
        self.smoothed_high_dev = np.array([self.high_dev[max(0, i-2):i+2].mean() for i in range(len(self.high_dev))])
        self.smoothed_dev = np.stack([self.smoothed_bass_dev, self.smoothed_mid_dev, self.smoothed_high_dev])


    def __len__(self):
        return len(self.bass)

    def visualize(self):
        from tqdm import tqdm
        x = np.arange(3)
        ylims1 = np.max(self.freqs)
        ylims2 = np.max(self.devs)
        for t in tqdm(range(len(self))):
            fig, ax1 = plt.subplots()
            ax1.plot(x, self.freqs[:, t], 'r')
            ax1.plot(x, self.smoothed[:, t], 'b')
            ax1.set_ylim(0, ylims1)
            ax1.set_ylabel('frequencies')
            ax2 = ax1.twinx()
            ax2.plot(x, self.devs[:, t], 'g')
            ax2.plot(x, self.smoothed_dev[:, t], 'y')
            ax2.set_ylim(0, ylims2)
            ax2.set_ylabel('std deviations')
            plt.savefig('fft_useful_%d.png' % t)
            plt.close()

    def draw_dominant_freqs(self):
        pass


class MusicalVariation(AnimatedVariation):
    def __init__(self, input_music, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.music = input_music
        self.scale = lambda x: np.log(np.abs(x + 0.001))
        self.t = 0

        weight_options = self.music.smoothed_all
        self.weight_dep = weight_options[:, np.random.choice(np.arange(weight_options.shape[1]))]

    def step(self):
        self.t += 1
        #print(self.music.smoothed_bass[self.t])
        for rf in list(self.anim_args[0:2]) + list(self.anim_args[3:5]):
            rf.deltas[0][0] = np.random.randn() * self.scale(self.music.smoothed_bass[self.t]) * (self.scale(self.music.smoothed_bass_dev[self.t]) ** 2)
            #rf.deltas[1][0] = -0.1 * np.random.randn() * self.scale(self.music.smoothed_bass_dev[self.t]) ** 2 * self.scale(self.music.smoothed_mid_dev[self.t]) ** 2 * self.scale(self.music.smoothed_high_dev[self.t]) ** 2
            #rf.deltas[2][0] = 3.0 * self.scale(self.music.smoothed_high[self.t]) * (self.scale(self.music.smoothed_high_dev[self.t]) ** 2)
            #print(rf.deltas)
            rf.deltas[0][1] = 0.0
            rf.deltas[1][1] = 0.0
            rf.deltas[2][1] = 0.0


        self.var.weight.deltas[0][1] = 1.0 + self.scale(self.weight_dep[self.t])
        #print(self.var.weight.deltas)

        super().step()