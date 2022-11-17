# BlockingCardAnalysis

Blocking cards are an affordable device for protecting smart cards. These devices are placed close to the smart cards and generate a noisy jamming signal or shield them.
Through this repo, we release the tools we developed for inspecting the spectrum emitted by blocking cards and setting up our attack against the MIFARE Classic and the MIFARE Ultralight.

## Prequisites

## Frequency Spectrum Analysis

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

### Demodulation

### Averaging

## Countermeasure
