#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
import librosa

import utils

# Compute local onset autocorrelation
DB = 'Ballroom'
GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]

# %% Q1
genres_p, genres_ALOTC = list(), list()

for g in GENRE[:1]:
    print('GENRE:', g)
    FILES = glob(DB + '/wav/' + g + '/*.wav')
    label, pred_t1, pred_t2, p_score, ALOTC_score = list(), list(), list(), list(), list()

    for f in FILES[:5]:
        f = f.replace('\\', '/')
        print('FILE:', f)

        # Read the labeled tempo
        bpm = float(utils.read_tempofile(DB, f))
        print('ground-truth tempo: ', bpm)
        label.append(bpm)

        # Estimate a static tempo
        sr, y = utils.read_wav(f)

        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

        # predict the tempo1(slower one), tempo2(faster one)
        # tempo1, tempo2 = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        tempo1, tempo2 = utils.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        pred_t1.append(tempo1)
        pred_t2.append(tempo2)
        print(tempo1, tempo2)

        # p score
        s1 = tempo1/(tempo1+tempo2)
        s2 = 1.0 - s1
        print(s1, s2)
        p = s1 * utils.P_score(tempo1, bpm) + s2 * utils.P_score(tempo2, bpm)
        p_score.append(p)

        # ALOTC score
        ALOTC = utils.ALOTC(tempo1, tempo2, bpm)
        ALOTC_score.append(ALOTC)

        print(p, ALOTC)

    p_avg = sum(p_score)/len(p_score)
    ALOTC_avg = sum(ALOTC_score)/len(ALOTC_score)
    genres_p.append(p_avg)
    genres_ALOTC.append(ALOTC_avg)

    print('----------')

print(genres_p)
print(genres_ALOTC)
print()

print("***** Q1 *****")
print("Genre          \tP-score    \tALOTC score")
for g in range(len(GENRE)):
    print("{:13s}\t{:8.2f}\t{:8.2f}".format(GENRE[g], genres_p[g], genres_ALOTC[g]))
print('----------')
print("Overall P-score:\t{:.2f}".format(sum(genres_p)/len(genres_p)))
print("Overall ALOTC score:\t{:.2f}".format(sum(genres_ALOTC)/len(genres_ALOTC)))

'''
[0.8213639886334281, 0.03636363636363636, 0.08497792805485113, 0.9865359865359866, 0.5166666666666667, 1.0, 0.5581395348837209, 0.0]
[0.826530612244898, 0.03636363636363636, 0.1076923076923077, 0.990990990990991, 0.5166666666666667, 1.0, 0.5581395348837209, 0.0]

***** Q1 *****
Genre          	P-score    	ALOTC score
Rumba        	    0.82	    0.83
Waltz        	    0.04	    0.04
VienneseWaltz	    0.08	    0.11
ChaChaCha    	    0.99	    0.99
Jive         	    0.52	    0.52
Tango        	    1.00	    1.00
Samba        	    0.56	    0.56
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.50
Overall ALOTC score:	0.50
'''