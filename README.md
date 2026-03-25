This repository provides the code used to generate the figures in [].

All data and figures are produced using Julia. Required packages include Struve, KrylovKit, FFTW, and ForwardDiff.

Fig 1 and Fig S1 are schematics with no corresponding datasets.

In folders corresponding to Fig 2, Fig 3, Fig S2, Fig S3, first run the corresponding fig#_data_generation.ipynb to generate the corresponding .csv data files, next run the corresponding fig#_plotter.ipynb to plot figures based on the .csv data.