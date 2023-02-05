# BlockingCardAnalysis

Blocking cards are an affordable device for protecting smart cards. These devices are placed close to the smart cards and generate a noisy jamming signal or shield them.
Through this repo, we release the tools we developed for inspecting the spectrum emitted by blocking cards and setting up our attack against the MIFARE Classic and the MIFARE Ultralight.

## Prequisites
In order to produce all results, a version of libnfc was used, in order to avoid segmentation faults: [https://github.com/blackwiz4rd/libnfc](https://github.com/blackwiz4rd/libnfc), refer to commit [#6de2cf2bb3d88fa70978363bde2eb7e490568f5d](https://github.com/blackwiz4rd/libnfc/commit/6de2cf2bb3d88fa70978363bde2eb7e490568f5d#r91875341).

## Frequency Spectrum Analysis
In order to evaluate the performance of the blocking cards, their spectrums were analyzed without card/reader interaction, just by activating the magnetic field:
* `Profiling.ipynb` : produces PDFs and PSDs of the blocking cards.
* `magnetic_field_up.c` : activates the EM field of the reader for <numSeconds>.

Usage:
```
magnetic_field_up <numSeconds>
```


## Attacking MIFARE Classic
The attack consists of two different phases: 
1. Recording the signal of the communication between the Reader and the MIFARE Classic
1. Demodulate the raw signal previously acquired.

### Msg Exchange
During the first phase, the attacker needs to capture the raw signals of the communication using ***GNURadio***. To let the reader send a specific sequence of commands to read data written on the MIFARE Classic, the attacker could rely on the ***mfclassic_apdu_get_data*** script, which can be used as follows:

```
mfclassic_apdu_get_data <numIterations>
```

where:

* `numIterations` : specify how many times the sequence of commands should be sent from the Reader to the MIFARE Classic.

### Demodulation
Once the signal has been collected, the attacker just need to launch one of the two following jupiter notebooks:
* `MifareClassicAtttack.ipynb` : use only one raw signal file in order to show entirely the communication between the Reader and the MIFARE Classic, decrypting all the messages that contain data retrieved from the MIFARE Classic.
* `DemodulationAnalysis.ipynb` : could use more raw signal files at the same time to analyze different blocking cards at once, by computing different metrics and storing them into a CSV file.

## Attacking MIFARE Ultralight

### Msg Exchange
During the first phase, the attacker needs to capture the raw signals of the communication using ***GNURadio***. To let the reader send a specific sequence of commands to read data written on the MIFARE Ultralight, the attacker could rely on the ***apdu_get_data*** script, which can be used as follows:

```
./apdu_get_data <numRepetitions> <numIterations>
```

where:
* `numRepetitions` : specifies the amount of repetitions for the experiment (e.g. the user can execute the experiment several times: the elecromagnetic field goes down from one experiment to the other).
* `numIterations` : specify how many times the sequence of commands should be sent from the Reader to the MIFARE Ultralight (e.g. 80). During iterations the electromagnetic field will be up.

### Demodulation
Once the signal has been collected, the attacker just need to launch one of the two following jupiter notebooks:
* `MifareUltralightASR.ipynb` : calculates attack success rate metrics.

### Averaging
For Mifare Ultralight analysis the averaging technique was used on multiple repetitions of the same signal portions for tag messages in order to evaluate if there are improvements in the decoding.

## Countermeasure
Two more jupiter notebooks are released to study the features that a noise emitted by a blocking card should have to be effective.
They both apply the same strategy:
1. Load the clean signal of the communication acquired using GNURadio.
1. Add noise to the clean signal in order to simulate the behaviour of a blocking card.
1. Analyze the performance of the demodulator in presence of different kinds of noise.

More in detail:
* `NoiseSimulationGaussian.ipynb` : Add gaussian noise at different percentage: 5%, 10%, 15%, 20%, 25%, 30%.
* `NoiseSimulationFixedFrequencies.ipynb` Add noise at different fixed frequencies.


# Images

#### GNURadio Schema
![My Image](imgs/GNURadioSchema.png)

### Spectrogram

#### BK1
![SpectrogramBK1](imgs/Spectrogram/bk1-1.png)
#### BK2
![SpectrogramBK2](imgs/Spectrogram/bk2-1.png)
#### BK3
![SpectrogramBK3](imgs/Spectrogram/bk3-1.png)
#### BK4
![SpectrogramBK4](imgs/Spectrogram/bk4-1.png)
#### BK5
![SpectrogramBK5](imgs/Spectrogram/bk5-1.png)
#### BK6
![SpectrogramBK6](imgs/Spectrogram/bk6-1.png)
#### BK7
![SpectrogramBK7](imgs/Spectrogram/bk7-1.png)
#### BK8
![SpectrogramBK8](imgs/Spectrogram/bk8-1.png)
#### BK9
![SpectrogramBK9](imgs/Spectrogram/bk9-1.png)
#### BK10
![SpectrogramBK10](imgs/Spectrogram/bk10-1.png)
#### BK11
![SpectrogramBK11](imgs/Spectrogram/bk11-1.png)

### PDF

#### BK1
![PdfBK1](imgs/Pdf/bk1-1.png)
#### BK2
![PdfBK2](imgs/Pdf/bk2-1.png)
#### BK3
![PdfBK3](imgs/Pdf/bk3-1.png)
#### BK4
![PdfBK4](imgs/Pdf/bk4-1.png)
#### BK5
![PdfBK5](imgs/Pdf/bk5-1.png)
#### BK6
![PdfBK6](imgs/Pdf/bk6-1.png)
#### BK7
![PdfBK7](imgs/Pdf/bk7-1.png)
#### BK8
![PdfBK8](imgs/Pdf/bk8-1.png)
#### BK9
![PdfBK9](imgs/Pdf/bk9-1.png)
#### BK10
![PdfBK10](imgs/Pdf/bk10-1.png)
#### BK11
![PdfBK11](imgs/Pdf/bk11-1.png)