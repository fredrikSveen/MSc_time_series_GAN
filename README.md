# time_series_GAN
This is the code related to my MSc thesis at the Norwegian University of Science and Technology (NTNU). The MSc program is Electronic system design and innovation with a specialization in signal processing. The goal of this thesis is to explore and compare the state of the art solutions for time series generation.

This MSc thesis is the next step of the project started in my specialization project, which took place during the Fall of 2023 and can be found [here](https://github.com/fredrikSveen/time_series_gan_project).

The following state-of-the-art methods are explored, but only TimeGAN is properly tested and discussed in the thesis. The other models listed below is models I believe could be suitable

|Generative model        | Capabilities                                            | Github link                                          |
|------------------------|---------------------------------------------------------|------------------------------------------------------|
|TimeGAN                 |Properly tested and discussed in the thesis              | [Link](https://github.com/fredrikSveen/TimeGAN)      |
|COSCI-GAN               |Has separate discriminators, and a central disciminator  | [Link](https://github.com/fredrikSveen/COSCI-GAN)    |
|Fourier Flows (+RealNVP)|Doing sequence generation in the frequency domain        | [Link](https://github.com/fredrikSveen/Fourier-flows)|
|GMMN                    |Not properly explored                                    | [Link](https://github.com/fredrikSveen/gmmn)         |

To compare TimeGAN to some of the more traditional methods, the methods listed below is also properly tested and discussed in the thesis

|Traditional method             | Description                                                    | File within this Github project                      |
|-------------------------------|----------------------------------------------------------------|------------------------------------------------------|
|Vectorized autoregressive (VAR)|A simple autoregressive method that can handle multivariate data|[AR_model.ipynb](https://github.com/fredrikSveen/MSc_time_series_GAN/blob/master/AR_model.ipynb)  |
|Basic RNN                      |A basic RNN (simpler than the GRU used in TimeGAN)              |[simple_rnn.ipynb](https://github.com/fredrikSveen/MSc_time_series_GAN/blob/master/simple_rnn.ipynb)|



### What are the different files?
|Filename                 |Description                                                        |
|-------------------------|-------------------------------------------------------------------|
|[AR_model.ipynb](AR_model.ipynb)|                                                            |
