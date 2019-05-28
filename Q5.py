#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import librosa
import mir_eval

import utils

# Compute local onset autocorrelation
# SMC or JCS
DB = 'JCS'
if DB == 'SMC':
    FILES = glob(DB + '/SMC_MIREX_Audio//*.wav')
elif DB == 'JCS':
    FILES = glob(DB + '/JCS_audio//*.wav')

# %% Q5
sum_f = 0.0
cnt_f = 0.0

for f in FILES:
    print('FILE:', f)

    # Read the labeled tempo
    g_beats = utils.read_beatfile(DB, f)
    # print('ground-truth beats:\n', g_beats)

    # Beat tracking
    sr, y = utils.read_wav(f)
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    timetag = librosa.frames_to_time(beats, sr=sr)
    # print('detect beats:\n', timetag)

    # F score
    f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
    print('f_measure:\n', f_measure)
    sum_f += f_measure
    cnt_f += 1.0


print('----------')

print("***** Q5 *****")
print("Database          \tF-score")
# SMC or JCS
print("{:13s}\t{:8.2f}".format('JCS', sum_f / cnt_f))