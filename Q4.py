#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import librosa
import mir_eval

import utils

# Compute local onset autocorrelation
DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q4
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

    genres_F.append(sum_f/cnt_f)

print('----------')
print(genres_F)

print("***** Q4 *****")
print("Genre          \tF-score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2f}".format(GENRE[g], genres_F[g]))
print('----------')
print("Overall F-score:\t{:.2f}".format(sum(genres_F)/len(genres_F)))