__author__ = 'oleg'
import os
from scipy.io import wavfile
from segmentation import Segmentation, sample2time, distance_to_next
from config import sound_dir, debug


def scan_folder(dirname, pitch_threshold=0.5):
    wavfilelist = [f for f in os.listdir(dirname) if os.path.splitext(f)[1] == '.wav']
    wavfilelist.sort()
    for wavfilename in wavfilelist:
        samplerate, signal = wavfile.read("{0}/{1}".format(dirname, wavfilename))
        pp = Segmentation(signal, samplerate, pitch_threshold)

        if debug:
            for i, maxlevel in enumerate(pp.maximum_levels):
                log = open("{0}/{1}_max_{2}.txt".format(dirname, wavfilename, i), "w")
                for x in maxlevel:
                    time = sample2time(samplerate, x)
                    log.write("{0} {1} {2}\n".format(time, time, pp.signal_filtered[x]))
                log.close()


            log = open("{0}/{1}_pp.txt".format(dirname, wavfilename), "w")
            for i, x in enumerate(pp.pitch_periods_start):
                time = sample2time(samplerate, x)
                log.write("{0} {1} {2}\n".format(time, time, 1000*sample2time(samplerate, distance_to_next(pp.pitch_periods_start, i))))
            log.close()

            log = open("{0}/{1}_pp_catched.txt".format(dirname, wavfilename), "w")
            for i, x in enumerate(pp.catched_shit):
                time = sample2time(samplerate, x)
                log.write("{0} {1} {2}\n".format(time, time, 1000*sample2time(samplerate, distance_to_next(pp.catched_shit, i))))
            log.close()


            log = open("{0}/{1}_max_pp.txt".format(dirname, wavfilename), "w")
            for x in pp.pitch_periods:
                time = sample2time(samplerate, x)
                log.write("{0} {1} {2}\n".format(time, time, pp.signal_filtered[x]))
            log.close()

            log = open("{0}/{1}_pp_transitions.txt".format(dirname, wavfilename), "w")
            for elem in pp.transitions:
                time = elem[0]
                log.write("{0} {1} {2}\n".format(time, time, elem[2]))
            log.close()

            log = open("{0}/{1}_pp_transitions2.txt".format(dirname, wavfilename), "w")
            for elem in pp.transitions2:
                time = elem[0]
                log.write("{0} {1} {2}\n".format(time, time, elem[2]))
            log.close()

            log = open("{0}/{1}_pp_transitions3.txt".format(dirname, wavfilename), "w")
            for elem in pp.transitions3:
                time = elem[0]
                log.write("{0} {1} {2}\n".format(time, time, elem[2]))
            log.close()

            log = open("{0}/{1}_pp_transitions4.txt".format(dirname, wavfilename), "w")
            for elem in pp.transitions4:
                time = elem[0]
                log.write("{0} {1} {2}\n".format(time, time, elem[2]))
            log.close()

            log = open("{0}/{1}_pp_transitions5.txt".format(dirname, wavfilename), "w")
            for elem in pp.transitions5:
                time = elem[0]
                log.write("{0} {1} {2}\n".format(time, time, elem[2]))
            log.close()

        log = open("{0}/{1}_segments.txt".format(dirname, wavfilename), "w")
        for elem in pp.segments:
            time = elem[0]
            log.write("{0} {1} {2}\n".format(time, time, elem[2]))
        log.close()

        print("{0} done... {1} segments found in {2} seconds.".format(wavfilename, len(pp.segments), pp.time))


def main():
    scan_folder(sound_dir, pitch_threshold=0.5)


if __name__ == "__main__":
    main()