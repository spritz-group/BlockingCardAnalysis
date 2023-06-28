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


### Team
Luca Attanasio (luca_attanasio@me.com)<br />
Marco Alecci (marco.alecci@uni.lu)<br />
Federico Turrin (turrin@math.unipd.it)<br />
Eleonora Losiouk (elosiouk@math.unipd.it)<br />
Alessandro Brighente (alessandro.brighente@unipd.it)<br />

We are members of [SPRITZ Security and Privacy Research Group](https://spritz.math.unipd.it/) at the University of Padua, Italy.

### Cite

Are you using OpenScope-sec in your research work? Please, cite us:
```bibtex   
The paper is still under submission
```
