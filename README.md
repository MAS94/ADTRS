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
If you find the code helpful in your research or work, please cite the following paper.
```
@INPROCEEDINGS{10647621,
  author={Alsaafin, Mohammed and Alsheikh, Musab and Anwar, Saeed and Usman, Muhammad},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Attention Down-Sampling Transformer, Relative Ranking and Self-Consistency For Blind Image Quality Assessment}, 
  year={2024},
  volume={},
  number={},
  pages={1260-1266},
  keywords={Image quality;Degradation;Adaptation models;Visualization;Image transformation;Transformers;Feature extraction;No-Reference Image Quality Assessment;CNNs;Transformers;Self-Consistency;Relative Ranking},
  doi={10.1109/ICIP51287.2024.10647621}}
```
