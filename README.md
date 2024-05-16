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
|Vectorized autoregressive (VAR)|A simple autoregressive method that can handle multivariate data|[AR_model.ipynb](AR_model.ipynb)    |
|Basic RNN                      |A basic RNN (simpler than the GRU used in TimeGAN)              |[simple_rnn.ipynb](simple_rnn.ipynb)|



### What are the different files?
|Filename                        |Description                                                       |
|--------------------------------|------------------------------------------------------------------|
|[AR_model.ipynb](AR_model.ipynb)| Testing auto_arima to find best ARIMA parameters, and use VAR to make multivariate forecast|
|[AR_script.py](AR_script.py)    | Script to run auto_arima on the server and test many different parameters                  |
|[curve_fitting.ipynb](curve_fitting.ipynb)  | Use curve-fitting to find the best fitting sine curve and the frequency. Find frequency deviation between estimate and true frequency and find RSS between the generated and fitted curve  |
|[curve_fitting_complex.ipynb](curve_fitting_complex.ipynb) | Same methods as in [curve_fitting.ipynb](curve_fitting.ipynb), but for a sine consisting of 2 sines added together |
|[data_generation.ipynb](data_generation.ipynb)| Notebook to generate sine waves and add additional sines on top.  |
|[data_loading.py](data_loading.py)| The methods used to generate the sines in [data_generation.ipynb](data_generation.ipynb), the data used in [timegan.py](timegan.py) and to generate data from a pretrained timegan-model  |
|[evaluation.ipynb](evaluation.ipynb)| This file is intended to collect all calculated metrics, but at the moment the different metrics are spread out in different files  |
|[generated_AR_sines.ipynb](generated_AR_sines.ipynb)|Showing the different sines generated by VAR   |
|[generated_complex_sines.ipynb](generated_complex_sines.ipynb)|Showing all the generated complex sines. NEED TO ADD VAR AND LSTM PLOTS   |
|[generated_timegan_sines.ipynb](generated_timegan_sines.ipynb)|Showing the simple sines generated by TimeGAN   |
|[power_spectrums.ipynb](power_spectrums.ipynb)|Using FFT to find the magnitude spectrum om the simple sines generated, and then using the spectrum to estimate the frequency of the generated sine   |
|[publicdata.py](publicdata.py)|File made to easily import idustrial data from Cognite OID   |
|[simple_rnn.ipynb](simple_rnn.ipynb)|Adaptation of a simple LSTM model to forecast data   |
|[statistical_evaluation.ipynb](statistical_evaluation.ipynb)|Statistical analysis of the generated data. Computation of KL and JS divergence|
|[timegan.py](timegan.py)| The model definition of TimeGAN  |
|[utils.py](utils.py)| Methods used i [timegan.py](timegan.py) and some general list processing functions. |
|[visual_analysis.ipynb](visual_analysis.ipynb)| Perform PCA and t-SNE on the generated data as a visual evaluation method  |   
