__author__ = 'oleg'

from scipy.signal import butter, lfilter, argrelextrema, medfilt, hamming
from scipy.stats.mstats import gmean
from scipy.spatial.distance import euclidean
import numpy as np
from bisect import bisect_left
import time
from config import *


#np.seterr(all='raise')
def sample2time(samplerate, sample_number):
    return 1.*sample_number/samplerate


def time2sample(samplerate, time):
    return int(time*samplerate)


def find_lt(a, x):
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    else:
        return 0


def find_lt_index(a, x):
    i = bisect_left(a, x)
    if i:
        return i-1
    else:
        return 0


def find_ge(a, x):
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    else:
        return a[-1]


def find_closest(a, x):
    try:
        lt = find_lt(a, x)
        ge = find_ge(a, x)
        if abs(x - lt) < abs(x - ge):
            return lt
        else:
            return ge
    except ValueError:
        return None


def normalize(signal):
    result = signal / np.max(np.abs(signal), axis=0)
    return result


def frame_energy(frame):
    if len(frame) == 0:
        return 0
    square_samples = np.array(frame, dtype=np.float64)**2
    energy = np.log(1+np.sqrt(np.mean(square_samples)))
    return energy


def signal_to_frames(signal, frame_size):
    markers = []
    cursor = 0
    while cursor < len(signal):
        markers.append(cursor)
        cursor += frame_size
    return markers


def find_zero_crossings(signal):
    return np.where(np.diff(np.sign(signal)))[0]


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class FilterSet:
    def __init__(self):
        cutfreqs = (100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7800)
        self.cutfreqs = zip(cutfreqs, cutfreqs[1:])
        self.channels = []

    def filter(self, samplerate, signal):
        channels = []
        for lowcut, highcut in self.cutfreqs:
            channels.append(butter_bandpass_filter(signal, lowcut, highcut, samplerate, 3))
        return channels


def side_func(transition, array, samplerate):
    threshold = 0.09*samplerate
    between_threshold = 0.03*samplerate
    to_prev = distance_to_prev(array, array.index(transition[0][1]))
    to_next = distance_to_next(array, array.index(transition[-1][1]))
    if len(transition) > 1:
        if to_prev > threshold and to_next > threshold and transition[-1][1]-transition[0][1] > between_threshold:
            return [transition[0], transition[-1]]
        elif to_prev > to_next:
            return [transition[0]]
        else:
            return [transition[-1]]
    else:
        return [transition[0]]


def pitch_length2frequency(samplerate, pitch_length):
    return samplerate/pitch_length


def find_maxima(signal):
    max_indices = argrelextrema(signal, np.greater)[0]
    max_values = np.array([signal[x] for x in max_indices])
    return max_indices, max_values


def find_all_maximum_levels(signal, number_of_levels=2):
    max_indices, max_values = find_maxima(signal)
    local_indices_levels = [max_indices]
    local_values_levels = [max_values]
    max_levels = []
    n = 1
    while True:
        local_max_indices, local_max_values = find_maxima(local_values_levels[-1])
        if n < number_of_levels and len(local_max_indices) > 6:
            local_indices_levels.append(local_max_indices)
            local_values_levels.append(local_max_values)
            n += 1
        else:
            break
    for i, level in enumerate(local_indices_levels):
        local_max_indices = level
        for j in range(i-1, -1, -1):
            local_max_indices = [local_indices_levels[j][x] for x in local_max_indices]
        max_levels.append(local_max_indices)
    return max_levels


def butter_lowpass(lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=2):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def flatness(signal):
    return gmean(signal) / np.mean(signal)


def flatness2(signal):
    signal = normalize(signal)
    mean = np.mean(signal)
    return sum([abs(i-mean) for i in signal])


def distance_to_next(array, index, default=1):
    try:
        return array[index+1] - array[index]
    except IndexError:
        return default


def distance_to_prev(array, index):
    try:
        return array[index] - array[index-1]
    except IndexError:
        return 1


def get_zcr(samplerate, signal, a, b):
    return len(find_zero_crossings(signal[a:b]))/sample2time(samplerate, len(signal[a:b]))


def compute_feature(args):
    x1, x2, channels, signal = args
    features = []
    energy = frame_energy(signal[x1:x2])
    zero = False
    if energy < 6:
        features = np.zeros(20)
        zero = True
    if not zero:
        for channel in channels:
            #features.append(frame_energy(channel[x1:x2]*hamming(len(channel[x1:x2]), sym=False)))
            features.append(frame_energy(channel[x1:x2]))
        features = np.array(features)
        features /= max(abs(features))
    return features


def compute_features_segmented(channels, markers, signal):
    markers = np.array(markers, dtype=int)
    newmarkers = [x+(channels, signal) for x in list(zip(markers, markers[1:]))+[(markers[-1], len(signal))]]
    result = [compute_feature(marker) for marker in newmarkers]
    return np.array(result)


def amplify(signal):
    return signal*(30500/max(signal))


def slope(signal, samplerate):
    return np.array([(signal[i+1]-signal[i])*samplerate for i in range(len(signal)-1)])


class Segmentation:
    def __init__(self, signal, samplerate, pitch_threshold=0.5):
        self.transitions = None
        self.channels = None
        start = time.time()
        self.signal = amplify(signal)
        self.signal2 = butter_bandpass_filter(self.signal, 120, 1000, samplerate, order=2)
        self.pitch_threshold = pitch_threshold
        self.samplerate = samplerate
        self.signal_filtered = butter_lowpass_filter(self.signal2, 500, self.samplerate, 2)
        self.signal_filtered = normalize(self.signal_filtered)
        self.maximum_levels = find_all_maximum_levels(self.signal_filtered, max_level)
        self.average_pitch = self.find_average_pitch(length=pitch_search_length)
        self.min_dist = self.average_pitch - self.average_pitch*self.pitch_threshold
        self.max_dist = self.average_pitch + self.average_pitch*self.pitch_threshold
        self.flat_distances_levels = self.find_flat_distances()
        self.pitch_periods = self.get_pitch_periods()
        self.catch_missed_periods()
        self.catch_missed_periods_backwards()
        self.periods_zcr = self.get_periods_zcr(self.pitch_periods, self.signal2)
        self.pitch_periods = self.clean_up(self.pitch_periods)
        self.catched_shit = np.array(self.pitch_periods)
        self.zero_crossings = find_zero_crossings(self.signal)
        self.zero_crossings_filtered = find_zero_crossings(self.signal_filtered)
        self.zeros_left_of_maxima = self.find_zeros_left_of_maxima()
        self.pitch_periods_start = self.find_pitch_periods_start()
        self.features = self.get_period_features(self.signal, self.pitch_periods_start)
        self.diffs = self.get_diffs()
        self.segments = self.find_segments(segmentation_threshold, self.pitch_periods_start)
        self.transitions2 = []
        self.transitions3 = []
        self.find_sounds_lost_in_vowels(0.1, 0.35, 4)
        self.find_sounds_lost_in_vowels(0.16, 0.25, 4)
        self.find_silent_markers(side="both")
        self.segments = self.segments_clean_up(self.segments)
        if check_tails:
            self.process_tails()
        self.transitions4 = []
        self.get_vowels_back()
        self.find_silent_markers(side="both")
        self.transitions5 = []
        self.find_consonant_borders()
        end = time.time()
        self.time = end-start

    def find_average_pitch(self, step=1, length=8, threshold=0.01):
        maxima = self.maximum_levels[-1]
        result = None
        number_of_steps = int(len(maxima)/step)
        i = 0
        while i < number_of_steps:

            if abs(1 - flatness([distance_to_next(maxima, x) for x in range(len(maxima))[i*step:i*step+length]])) < threshold:

                result = int(np.mean([distance_to_next(maxima, x) for x in range(i*step, i*step+length)]))
                break
            i += 1
        return result

    def find_flat_distances(self, step=1, length=3, threshold=0.02):
        min_distance = self.min_dist
        flat_distances = []
        for level in self.maximum_levels:
            level_flat_distances = np.zeros(len(level), dtype=np.int)
            number_of_steps = int(len(level)/step)-1
            i = 0
            while i < number_of_steps:
                if abs(1 - flatness([distance_to_next(level, x) for x in range(i*step, i*step+length)])) < threshold:
                    if min_distance <= min([distance_to_next(level, x) for x in range(i*step, i*step+length)]):
                        level_flat_distances[i*step: i*step+length] = [1]*length
                i += 1
            flat_distances.append(level_flat_distances)
        return flat_distances

    def get_pitch_periods(self):
        pitch_periods = []
        for level, flat_distances in reversed(list(zip(self.maximum_levels, self.flat_distances_levels))):
            flat_level = []
            for i, is_flat in zip(level, flat_distances):
                if is_flat:
                    flat_level.append(i)
            if len(pitch_periods) == 0:
                pitch_periods += flat_level
            else:
                for item in flat_level:
                    if not item in pitch_periods and find_ge(pitch_periods, item) - find_lt(pitch_periods, item) > self.max_dist:
                        pitch_periods.append(item)
                        pitch_periods.sort()
        return pitch_periods

    # def get_pitch_periods(self):
    #     pitch_periods = []
    #     for level, flat_distances in reversed(list(zip(self.maximum_levels, self.flat_distances_levels))):
    #         flat_level = []
    #         for i, is_flat in zip(level, flat_distances):
    #             if is_flat:
    #                 flat_level.append(i)
    #         if len(pitch_periods) == 0:
    #             pitch_periods += flat_level
    #         else:
    #             for item in flat_level:
    #                 if not item in pitch_periods and find_ge(pitch_periods, item) - find_lt(pitch_periods, item) > self.max_dist:
    #                     pitch_periods.insert(find_lt_index(pitch_periods, item), item)
    #                     #pitch_periods.append(item)
    #                     #pitch_periods.sort()
    #     return pitch_periods

    def catch_missed_periods(self):
        i = 0
        changed = 0
        length = len(self.pitch_periods)
        while i < length - 1:
            if not self.min_dist < distance_to_next(self.pitch_periods, i) < self.max_dist:
                closest = find_closest(self.maximum_levels[0], self.pitch_periods[i]+distance_to_next(self.pitch_periods, i-1))
                if closest:
                    dist = abs(closest - self.pitch_periods[i])
                    if not closest in self.pitch_periods and self.min_dist < dist < self.max_dist and not self.is_unvoiced(self.signal2, closest-self.average_pitch/2, closest+self.average_pitch/2):
                        self.pitch_periods.append(closest)
                        changed = 1
            i += 1
        self.pitch_periods.sort()
        if changed:
            self.catch_missed_periods()

    def catch_missed_periods_backwards(self):
        i = len(self.pitch_periods)-1
        changed = 0
        while i > 0:
            if not self.min_dist < distance_to_next(self.pitch_periods, i-1) < self.max_dist:
                closest = find_closest(self.maximum_levels[0], self.pitch_periods[i]-distance_to_next(self.pitch_periods, i))
                if closest:
                    dist = abs(closest - self.pitch_periods[i])
                    if not closest in self.pitch_periods and self.min_dist < dist < self.max_dist and not self.is_unvoiced(self.signal2, closest-self.average_pitch/2, closest+self.average_pitch/2):
                        self.pitch_periods.append(closest)
                        changed = 1
            i -= 1
        self.pitch_periods.sort()
        if changed:
            self.catch_missed_periods_backwards()

    def find_zeros_left_of_maxima(self):
        zlo = []
        for x in self.pitch_periods:
            zero_index = find_lt(self.zero_crossings_filtered, x)
            if not zero_index in zlo:
                zlo.append(zero_index)
        return zlo

    def find_pitch_periods_start(self):
        fpps = [0.]
        for x in self.zeros_left_of_maxima:
            try:
                zero_index = find_lt(self.zero_crossings, x)
                if not zero_index in fpps:
                    fpps.append(zero_index)
            except ValueError:
                pass
        return fpps

    def is_unvoiced(self, signal, a, b):
        zcr = get_zcr(self.samplerate, signal, a, b)
        amp = frame_energy(signal[a:b])
        if zcr > zcr_threshold or amp < unvoiced_check_amp_threshold:
            return True

    def clean_up(self, array):
        to_clean = []
        for i, x in enumerate(array[:-1]):
            if distance_to_next(array, i) < self.min_dist or self.periods_zcr[i] > zcr_threshold or frame_energy(self.signal2[array[i]:array[i+1]]) < cleanup_amp_threshold:
                to_clean.append(i+1)
        for i in to_clean[::-1]:
            array.pop(i)
        return array

    def segments_clean_up(self, array):
        to_clean = []
        t, xs, z, x = zip(*array)
        for i, x in enumerate(array[:-1]):
            if frame_energy(self.signal[array[i][1]:array[i+1][1]]) < 6 or distance_to_next(xs, i) < self.average_pitch*2:
                to_clean.append(i)
        for i in to_clean[::-1]:
            array.pop(i)
        return array

    def get_pitch_frequency(self):
        pitch_frequency_array = []
        for i, x in enumerate(self.pitch_periods_start[:-1]):
            pitch_frequency_array.append(pitch_length2frequency(self.samplerate, distance_to_next(self.pitch_periods_start, i)))
        pitch_frequency_array = medfilt(pitch_frequency_array)
        return pitch_frequency_array

    # def get_pitch_between(self, a, b):
    #     index = self.pitch_periods_start.index(find_closest(self.pitch_periods_start, (a+b)/2))
    #     try:
    #         return self.pitch_frequency[index]
    #     except IndexError:
    #         return 0

    def get_periods_zcr(self, periods, signal):
        periods_zcr = []
        for i, x in enumerate(periods[:-1]):
            periods_zcr.append(len(find_zero_crossings(signal[int(x):periods[i+1]]))/sample2time(self.samplerate, periods[i+1]-x))
        return periods_zcr

    def get_period_features(self, signal, periods):
        channels = FilterSet().filter(self.samplerate, signal)
        features = compute_features_segmented(channels, periods, signal)
        return features

    def get_diffs(self):
        if diff_method == "double":
            result = [0, 0]
        else:
            result = [0]
        features = self.features
        if diff_method == "around":
            for i, feature in enumerate(features[1:-1]):
                diff = euclidean(features[i-1], features[i+1])
                result.append(diff)
        elif diff_method == "double":
            for i, feature in enumerate(features[2:-2]):
                diff1 = euclidean(features[i], features[i-2])
                diff2 = euclidean(features[i], features[i-1])
                diff3 = euclidean(features[i], features[i+1])
                diff4 = euclidean(features[i], features[i+2])
                result.append(diff1 + diff2 + diff3 + diff4)
        else:
            for i, feature in enumerate(features[1:-1]):
                diff1 = euclidean(features[i], features[i-1])
                diff2 = euclidean(features[i], features[i+1])
                result.append(diff1 + diff2)
        result.append(result[-1])
        if diff_method == "double":
            result.append(result[-1])
        return result

    def find_segments(self, threshold, periods):
        step1 = []
        step1index = 0
        for i, x in enumerate(periods):
            time = sample2time(self.samplerate, x)
            dist = self.diffs[i]
            if distance_to_next(periods, i, default=self.average_pitch*2.1) > self.average_pitch*2 or distance_to_prev(periods, i) > self.average_pitch*2:
                dist += 1
            if dist > threshold:
                step1.append((time, x, dist, step1index))
                step1index += 1
        times, xs, dists, indices = zip(*step1)
        sequences = []
        current_sequence = []
        for i, elem in enumerate(step1):
            current_sequence.append(elem)
            if distance_to_next(xs, i, default=self.average_pitch*2) > self.average_pitch*1.835:
                sequences.append(current_sequence)
                current_sequence = []

        step2 = [(0, 0, 0, 0)]
        for sequence in sequences:
            if len(sequence) > 1:
                if no_first_elem:
                    if sequence[0][2] > first_elem_threshold and distance_to_prev(xs, sequence[0][3]) > self.average_pitch*2:
                        sequence.pop(0)
                step2.extend(side_func((sequence[0], sequence[-1]), xs, self.samplerate))
            else:
                if no_first_elem:
                    if sequence[0][2] < first_elem_threshold:
                        step2.append(sequence[0])
                else:
                    step2.append(sequence[0])
        self.transitions = step1
        return step2

    def find_silent_markers(self, side="both"):
        times, frames, d, i = zip(*self.segments)
        frame_size = frame_size_sec*self.samplerate
        for i, elem in enumerate(self.segments[:-1]):
            energy = frame_energy(self.signal[frames[i]:frames[i+1]])
            if sample2time(self.samplerate, distance_to_next(frames, i)) > 0.1 and energy < 9:
                tempsignal = self.signal[frames[i]:frames[i+1]]
                tempframes = signal_to_frames(tempsignal, frame_size)
                tempframes = np.array(tempframes, dtype=int)
                amps = [(j, frame_energy(tempsignal[tempframes[j]:tempframes[j+1]])) for j, time in enumerate(tempframes[:-1])]
                amps += [(len(amps), frame_energy(tempsignal[tempframes[-1]:len(tempsignal)-1]))]
                newmarkers = []
                prev = amps[0]
                for amp in amps[1:]:
                    if side != "right" and amp[1] < silence_markers_threshold and prev[1] > silence_markers_threshold:
                        newmarkers.append(amp[0])
                    if side != "left" and amp[1] > silence_markers_threshold and prev[1] < silence_markers_threshold:
                        newmarkers.append(amp[0])
                    prev = amp
                for newmarker in newmarkers:
                    self.segments.append((sample2time(self.samplerate, frames[i]+frame_size*newmarker), frames[i]+frame_size*newmarker, 0, 0))

        self.segments.sort(key=lambda x: x[1])

    def find_sounds_lost_in_vowels(self, length, threshold, mp):
        times, frames, dist, indices = zip(*self.segments)
        frames = np.array(frames, dtype=int)
        for i, elem in enumerate(self.segments[:-1]):
            energy = frame_energy(self.signal[frames[i]:frames[i+1]])
            if sample2time(self.samplerate, distance_to_next(frames, i)) > length and energy > 9:
                start = self.pitch_periods_start.index(frames[i])
                end = self.pitch_periods_start.index(frames[i+1])
                periods = self.pitch_periods_start[start:end+1]

                step1 = []
                step1index = 0
                for j, x in enumerate(periods):
                    time = sample2time(self.samplerate, self.pitch_periods_start[start+j])
                    dist = self.diffs[start+j]
                    if dist > threshold:
                        step1.append((time, x, dist, j))
                        self.transitions2.append((time, x, dist, j))
                        step1index += 1
                if len(step1) == 0:
                    return 0
                times, xs, dists, indices = zip(*step1)

                step2 = []
                for j, e in enumerate(step1):
                    if (distance_to_next(xs, j) > self.average_pitch*mp and self.pitch_periods_start[start+e[3]] - self.pitch_periods_start[start] > self.average_pitch*mp) or (distance_to_prev(xs, j) > self.average_pitch*mp and self.pitch_periods_start[end] - self.pitch_periods_start[start+e[3]] > self.average_pitch*mp):
                        step2.append(e)
                        self.transitions3.append(e)

                step2 = [(sample2time(self.samplerate, self.pitch_periods_start[start]), self.pitch_periods_start[start], 0, 0)]+step2+[(sample2time(self.samplerate, self.pitch_periods_start[end]), self.pitch_periods_start[end], 0, 0)]
                times, xs, dists, indices = zip(*step2)

                step3 = []
                for j, e in enumerate(step2):
                    #if distance_to_next(xs, j) > self.average_pitch*(mp/2) and distance_to_prev(xs, j) > self.average_pitch*(mp):
                    if distance_to_prev(xs, j) > self.average_pitch*mp or distance_to_next(xs, j) > self.average_pitch*mp:
                        step3.append(e)
                times, xs, dists, indices = zip(*step3)

                step4 = []
                for j, e in enumerate(step3):
                    if distance_to_next(xs, j) > self.average_pitch*mp:
                        step4.append(e)

                t, f, x, z = zip(*self.segments)
                for e in step4:
                    if not e[1] in f:
                        self.segments.append(e)
        self.segments.sort(key=lambda x: x[1])

    def process_tails(self):
        tails = [segment for segment in self.segments if segment[2] == 0]
        t, f, x, z = zip(*self.segments)
        to_del = []
        features = self.get_period_features(self.signal, f)
        for tail in tails:
            index = f.index(tail[1])
            features1 = features[index]
            features2 = features[index+1]
            dist = euclidean(features1, features2)*2

            zcr1 = get_zcr(self.samplerate, self.signal2, f[index], f[index+1])
            try:
                zcr2 = get_zcr(self.samplerate, self.signal2, f[index+1], f[index+2])
            except IndexError:
                zcr2 = get_zcr(self.samplerate, self.signal2, f[index+1], len(self.signal2)-1)
            amp1 = frame_energy(self.signal2[f[index]: f[index+1]])
            try:
                amp2 = frame_energy(self.signal2[f[index+1]: f[index+2]])
            except IndexError:
                amp2 = frame_energy(self.signal2[f[index+1]: len(self.signal2)-1])
            maxzcr = max(zcr1, zcr2)
            zcrdist = abs(zcr2/maxzcr-zcr1/maxzcr)
            dist += zcrdist
            dist += amp2-amp1
            #print(tail, dist)
            #print(sample2time(self.samplerate, f[index]), dist, amp2-amp1)
            if dist < 2.2 and amp2 > amp1:
                to_del.append(index+1)
        for i in to_del[::-1]:
            self.segments.pop(i)

    def get_vowels_back(self):
        t, f, x, z = zip(*self.segments)
        to_del = []
        features = self.get_period_features(self.signal, f)
        for i, segment in enumerate(self.segments[:-2]):
            energy1 = frame_energy(self.signal[f[i]:f[i+1]])
            energy2 = frame_energy(self.signal[f[i+1]:f[i+2]])
            if energy1 > 9 and energy2 > 9:
                zcr1 = get_zcr(self.samplerate, self.signal, f[i], f[i+1])
                zcr2 = get_zcr(self.samplerate, self.signal, f[i+1], f[i+2])
                max_zcr = max(zcr1, zcr2)
                features1 = np.concatenate((features[i], [zcr1/max_zcr]))
                features2 = np.concatenate((features[i+1], [zcr2/max_zcr]))
                diff = euclidean(features1, features2)
                if diff < 0.33:
                    to_del.append(i+1)
                self.transitions4.append((t[i], f[i], diff, z[i]))
        for i in to_del[::-1]:
            self.segments.pop(i)

    def find_consonant_borders(self):
        t, f, x, z = zip(*self.segments)
        mp = 6
        finalj = None
        frame_size = frame_size_sec*self.samplerate
        length = 0.16*self.samplerate
        for i, segment in enumerate(self.segments[:-1]):
            if f[i+1]-f[i] > length:
                zcr = get_zcr(self.samplerate, self.signal, f[i], f[i+1])
                amp = frame_energy(self.signal[f[i]: f[i+1]])
                if zcr > 1500 and amp > 8:
                    tempsignal = self.signal[f[i]:f[i+1]]
                    tempframes = signal_to_frames(tempsignal, frame_size)
                    tempframes = np.array(tempframes, dtype=int)
                    for j, feature in zip(range(2, len(tempframes)-3), tempframes[2:-2]):
                        zcr1 = get_zcr(self.samplerate, tempsignal, tempframes[j-2], tempframes[j-1])
                        zcr2 = get_zcr(self.samplerate, tempsignal, tempframes[j+2], tempframes[j+3])
                        max_zcr = max(zcr1, zcr2)
                        diff = abs(zcr2/max_zcr-zcr1/max_zcr)
                        self.transitions5.append((sample2time(self.samplerate, f[i]+frame_size*j), f[i]+frame_size*j, get_zcr(self.samplerate, tempsignal, tempframes[j], tempframes[j+1]), z[i]))
                        if diff > 0.7 and ((f[i]+j*frame_size - f[i] > self.average_pitch*mp) and (f[i+1] - (f[i]+j*frame_size) > self.average_pitch*mp)):
                            finalj = j

                    if finalj:
                        self.segments.append((sample2time(self.samplerate, f[i]+frame_size*finalj), f[i]+frame_size*finalj, diff, z[i]))
        self.segments.sort(key=lambda x: x[1])

