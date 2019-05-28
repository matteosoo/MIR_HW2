#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import librosa
import madmom
import mir_eval

import utils

# Compute local onset autocorrelation
# SMC or JCS
DB = 'JCS'
if DB == 'SMC':
    FILES = glob(DB + '/SMC_MIREX_Audio//*.wav')
elif DB == 'JCS':
    FILES = glob(DB + '/JCS_audio//*.wav')

# %% Q6
sum_f = 0.0
cnt_f = 0.0

for f in FILES:
    print('FILE:', f)

    # Read the labeled tempo
    g_beats = utils.read_beatfile(DB, f)
    # print('ground-truth beats:\n', g_beats)

    # madmom
    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    # Beat tracking
    act = madmom.features.beats.RNNBeatProcessor()(f)
    timetag = proc(act)

    # F score
    f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
    print('f_measure:\n', f_measure)
    sum_f += f_measure
    cnt_f += 1.0

print('----------')

print("***** Q6 *****")
print("Database          \tF-score")
# SMC or JCS
print("{:13s}\t{:8.2f}".format('JCS', sum_f / cnt_f))