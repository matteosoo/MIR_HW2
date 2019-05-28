#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import madmom

import utils

# Compute local onset autocorrelation
DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q3
genres_p, genres_ALOTC = list(), list()

for g in GENRE:
    print('GENRE:', g)
    FILES = glob(DB + '/wav/' + g + '/*.wav')
    label, pred_t1, pred_t2, p_score, ALOTC_score = list(), list(), list(), list(), list()

    for f in FILES:
        f = f.replace('\\', '/')
        print('FILE:', f)

        # Read the labeled tempo
        bpm = float(utils.read_tempofile(DB, f))
        print('ground-truth tempo: ', bpm)
        label.append(bpm)

        # madmom estimate tempo
        proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(f)
        tempo1 = (proc(act)).astype(float)[0][0].item()
        tempo2 = (proc(act)).astype(float)[1][0].item()
        print(tempo1, tempo2)

        # p score
        s1 = tempo1 / (tempo1 + tempo2)
        s2 = 1.0 - s1
        print(s1, s2)
        p = s1 * utils.P_score(tempo1, bpm) + s2 * utils.P_score(tempo2, bpm)
        p_score.append(p)

        # ALOTC score
        ALOTC = utils.ALOTC(tempo1, tempo2, bpm)
        ALOTC_score.append(ALOTC)

        print(p, ALOTC)

    p_avg = sum(p_score) / len(p_score)
    ALOTC_avg = sum(ALOTC_score) / len(ALOTC_score)
    genres_p.append(p_avg)
    genres_ALOTC.append(ALOTC_avg)

    print('----------')

print(genres_p)
print(genres_ALOTC)
print()

print("***** Q3 *****")
print("Genre          \tP-score    \tALOTC score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2f}\t{:8.2f}".format(GENRE[g], genres_p[g], genres_ALOTC[g]))
print('----------')
print("Overall P-score:\t{:.2f}".format(sum(genres_p) / len(genres_p)))
print("Overall ALOTC score:\t{:.2f}".format(sum(genres_ALOTC) / len(genres_ALOTC)))