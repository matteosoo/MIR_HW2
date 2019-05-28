#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import madmom
import mir_eval

import utils

# Compute local onset autocorrelation
DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q6-ballroom
genres_F = list()

for g in GENRE:
    print('GENRE:', g)
    FILES = glob(DB + '/wav/' + g + '/*.wav')
    sum_f = 0.0
    cnt_f = 0.0

    for f in FILES:
        f = f.replace('\\', '/')
        print('FILE:', f)

        # Read the labeled tempo
        g_beats = utils.read_beatfile(DB, f)
        # print('ground-truth beats:\n', g_beats)

        # madmom beat tracking
        proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(f)
        timetag = proc(act)

        # F score
        f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
        print('f_measure:\n', f_measure)
        sum_f += f_measure
        cnt_f += 1.0

    genres_F.append(sum_f / cnt_f)

print('----------')
print(genres_F)

print("***** Q6-Ballroom *****")
print("Genre          \tF-score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2f}".format(GENRE[g], genres_F[g]))
print('----------')
print("Overall F-score:\t{:.2f}".format(sum(genres_F)/len(genres_F)))