This repository provides the code used to generate the figures in [].

All data and figures are produced using Julia in jupyter notebook. Required packages include Struve, KrylovKit, FFTW, and ForwardDiff. Multi-threading is used to speed-up the code generating Fig 3 and Fig S2, so please open jupyter notebook in terminal with "JULIA_NUM_THREADS=4 jupyter notebook".

Fig 1 and Fig S1 are schematics with no corresponding datasets.

In folders corresponding to Fig 2, Fig 3, Fig S2, Fig S3, first run the corresponding fig#_data_generation.ipynb to generate the corresponding .csv data files, next run the corresponding fig#_plotter.ipynb to plot figures based on the .csv data.