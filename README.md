# ADTRS
The source code for the ICIP 2024 paper titled 'Attention Down-Sampling Transformer, Relative Ranking, and Self-Consistency for Blind Image Quality Assessment'.

## Requirements

The model is built using
- Python 3.7
- PyTorch 1.7.0+cu110
- TorchVision 0.8.0
- scipy
- numpy
## Datasets
In this study, we utilize five datasets for evaluation ([LIVE]( https://live.ece.utexas.edu/research/quality/subjective.htm), [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/), [TID2013](http://www.ponomarenko.info/tid2013.htm), [CSIQ](https://s2.smu.edu/~eclarson/csiq.html), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html))

In the run.py file, update the folderpath to match the name of the dataset folder within the project directory.

## Acknowledgement
This code incorporates components from [TReS](https://github.com/isalirezag/TReS) and [EfficientFormer](https://github.com/snap-research/EfficientFormer). 

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
Will be updated once the final version is published and a DOI is assigned.
```
