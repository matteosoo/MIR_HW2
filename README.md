# MIR_HW2
# Outline

Task 1: tempo estimation

Task 2: using dynamic programming for beat tracking

Task 3: meter recognition

# Task 1: tempo estimation
## Q1
* 說明：
在Ballroom這套dataset中，由於整體的資料集都偏向於「舞曲」的形式，故大體上來說在tempo上的預測是不錯的。
就細項看來，如“ChaChaCha”, “Tango”都看來是較佳的結果，但我們也不能因此說它就是最適合被預測的節奏，因為不同參數表現下將可能有不同表現，我們只能說在這樣的參數設定下是適合這樣類型的資料集。
另外，P score低於ALOTC score的結果也是可以被預期的，因為ALOTC是採兩個predict的tempo值之一的落在接受範圍內即對的方式運算。

* 結果：

**P score:**
[0.5081151031247536, 0.5692655367570308, 0.048749463368417935, 0.990990990990991, 0.10769895031859616, 0.7891538637927082, 1.0, 0.0]
**ALOTC score:**
[0.7209302325581395, 0.65, 0.07272727272727272, 0.990990990990991, 0.13846153846153847, 0.8163265306122449, 1.0, 0.0]

```python=
***** Q1 *****
Genre             P-score    	ALOTC score
Samba        	    0.51	    0.72
Jive         	    0.57	    0.65
Waltz        	    0.05	    0.07
ChaChaCha    	    0.99	    0.99
VienneseWaltz	    0.11	    0.14
Rumba        	    0.79	    0.82
Tango        	    1.00	    1.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.50
Overall ALOTC score:	0.55
```
* 備註：
上欄[]陣列內數值串為程式運算結果，而下方整理於程式碼欄的資訊為四捨五入取小數後第2位的結果。

## Q2
* 說明：
由於tempo很容易受到harmonic的影響，故嘗試了不同的計算方式避免偏見，也可以藉此找出ballroom中各種曲風(genre)類型的dataset是否具特定的特徵。
從以下4.大點的運算結果觀察，在原先運算不凸顯的score有機會在調整不同的參數下，脫穎而出。例如Quickstep在乘以兩倍(*2) temp後就從原先 (p score, ALOTC score)為 (0,0)拉升到 (0.92,0.92)的表現。

* 結果
### 1. [𝑇 1 /2, 𝑇 2/2]

**P score:**
[0.030264415418923222, 0.0, 0.13410974723114782, 0.0, 0.0, 0.10880531988076131, 0.0, 0.0]
**ALOTC score:**
[0.03488372093023256, 0.0, 0.15454545454545454, 0.0, 0.0, 0.12244897959183673, 0.0, 0.0]

```python=
***** Q2 *****
Genre             P-score    	ALOTC score
Samba        	    0.03	    0.03
Jive         	    0.00	    0.00
Waltz        	    0.13	    0.15
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Rumba        	    0.11	    0.12
Tango        	    0.00	    0.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.03
Overall ALOTC score:	0.04
```

### 2. [𝑇 1/3, 𝑇 2/3]

**P score:**
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
**ALOTC score:**
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

```python=
***** Q2 *****
Genre             P-score    	ALOTC score
Samba        	    0.00	    0.00
Jive         	    0.00	    0.00
Waltz        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Rumba        	    0.00	    0.00
Tango        	    0.00	    0.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.00
Overall ALOTC score:	0.00
```
### 3. [2𝑇 1 , 2𝑇 2]

**P score:**
[0.0, 0.22178807773151735, 0.0, 0.0, 0.10792610838338562, 0.0, 0.0, 0.9202014846235419]
**ALOTC score:**
[0.0, 0.43333333333333335, 0.0, 0.0, 0.13846153846153847, 0.0, 0.0, 0.926829268292683]


```python=
***** Q2 *****
Genre             P-score    	ALOTC score
Samba        	    0.00	    0.00
Jive         	    0.22	    0.43
Waltz        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.11	    0.14
Rumba        	    0.00	    0.00
Tango        	    0.00	    0.00
Quickstep    	    0.92	    0.93
----------
Overall P-score:	0.16
Overall ALOTC score:	0.19
```
### 4. [3𝑇 1 , 3𝑇 2]

**P score:**
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
**ALOTC score:**
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


```python=
***** Q2 *****
Genre             P-score    	ALOTC score
Samba        	    0.00	    0.00
Jive         	    0.00	    0.00
Waltz        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Rumba        	    0.00	    0.00
Tango        	    0.00	    0.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.00
Overall ALOTC score:	0.00
```

## Q3
* 說明：
這邊使用MADMOM這個library作tempo預測，其中的*madmom.features.tempo.TempoEstimationProcessor*可以利於抓出一連串由高至低可能的tempo, 再同樣透過只抓取前兩名的tempo預測結果，作兩種score的運算。
而madmom藉由RNN的神經網路運算強化，使它目前在結果上，比起librosa的運算結果還要來的更佳。

* 結果：

MADMOM:

**P score:**
[0.5085387287848985, 0.5261774321616962, 0.696423676965648, 0.6466456383185242, 0.6667957158699837, 0.6728964566450605, 0.5575845360942514, 0.5209321297486066]
**ALOTC score:**
[0.9489795918367347, 0.9636363636363636, 1.0, 0.990990990990991, 1.0, 1.0, 0.9651162790697675, 0.7804878048780488]

```python=
***** Q3 *****
Genre             P-score    	ALOTC score
Rumba        	    0.51	    0.95
Waltz        	    0.53	    0.96
VienneseWaltz	    0.70	    1.00
ChaChaCha    	    0.65	    0.99
Jive         	    0.67	    1.00
Tango        	    0.67	    1.00
Samba        	    0.56	    0.97
Quickstep    	    0.52	    0.78
----------
Overall P-score:	0.60
Overall ALOTC score:	0.96
```


# Task 2: using dynamic programming for beat tracking

## Q4
* 說明：
*mireval.beat* 在協助F score的運算上有莫大幫忙，也能支持與ground truth資料的0.07秒緩衝容錯，真的必須大力推崇。

* 結果：

Ballroom dataset: 

**F score:**
[0.5446045250558064, 0.6150824788481358, 0.5450631515132844, 0.820329195256893, 0.7115361639888207, 0.7305974462707833, 0.7447967742403079, 0.547331101393421]

```python=
***** Q4 *****
Genre             F-score
Samba        	    0.54
Jive         	    0.62
Waltz        	    0.55
ChaChaCha    	    0.82
VienneseWaltz	    0.71
Rumba        	    0.73
Tango        	    0.74
Quickstep    	    0.55
----------
Overall F-score:	0.66
```
## Q5
* 說明：
Ballroom 與以下兩個datasets相較來說結果較佳，最大原因應該還是在於它的資料集本身是以「舞曲」為主。
SMC 應該是相較難的資料集，因為它當中的音樂多半是古典音樂、抒情歌..等，旋律線條大過於節奏性的音樂，所以很容易被旋律線拉著跑。
JCS 其實也有節奏性強烈的歌曲，但由於融合了一些變換節奏，我猜測這可能是使他相較Ballroom略為偏低的理由。

* 結果：

SMC and JCS dataset: 

```python=
***** Q5 *****
Database          F-score
SMC          	    0.36
JCS          	    0.63
```
## Q6
* 說明：
Madmom作為目前最佳結果表現之一的library，約莫可相較前面Q4, Q5 提升0.15~0.20的F score表現。

* 結果：

### 1. Beat tracking
* for the SMC and the JCS dataset
```python=
***** Q6 *****
Database          F-score
SMC          	    0.52
JCS          	    0.82
```
* for the Ballroom dataset

**F score**
[0.8541896120988937, 0.8340432332403245, 0.8163842729678702, 0.8743322780139349, 0.8528222277676363, 0.8270410218567891, 0.866726889695503, 0.6784941085309891]
```python=
***** Q6-Ballroom *****
Genre             F-score
Samba        	    0.85
Jive         	    0.83
Waltz        	    0.82
ChaChaCha    	    0.87
VienneseWaltz	    0.85
Rumba        	    0.83
Tango        	    0.87
Quickstep    	    0.68
----------
Overall F-score:	0.83
```

### 2. Downbeat tracking
* for the Ballroom and the JCS dataset
```python=
***** Q6 *****
Database          F-score
JCS         	    0.72
Ballroom    	    0.74
```

# Task 3: meter recognition
## Q7
### IDEA
針對音樂上這樣[複節拍（polymeter）](https://zh.wikipedia.org/wiki/节拍#复节拍（混合节拍）)，又或是會被俗稱「變態拍」這部分，設計演算法上最重要應該在於掌握它的動態變化。

而首先前處理上，可以參考mireval官方library中的文件有提到，利用“*mireval.beat.trimbeats*”作為data preprocessing去除音樂前五秒的片段以利提升資料的正確性。

此外，適當的去除開頭的「弱起拍」節奏，以及消除尾巴不完整的小節拍數，將可有效從正確位置上開始被偵測到正確結束。

再者，必須分析tempogram中onset的downbeat時間點，例如[0, 3, 7 ,10]並假設抓到的結尾時間點為[15]，如此一來我們將可以將list中相鄰位置相減，得到[3, 4, 3, 5]這樣的動態beats_per_bar的變化關係。

最後配合madmom中的*madmom.features.DBNDownBeatTrackingProcessor(beatsperbar=beatperbar, fps=100)* 餵入預測。

---
# Code link 🔗
> Code here: https://github.com/matteosoo/MIR_HW2
