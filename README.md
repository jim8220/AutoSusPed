# AutoSusPed
  Research in automatic piano music generation has only recently started to involve piano pedals as a part of the task. In this work, we train various neural network architectures with piano sustain pedal control-change (CC) data using different categories within the MAESTRO classical piano music dataset to study the performances of basic models and test the suitability of neural networks in an automatic piano pedal styling task. By changing the temporal scanning range of convolution kernels and the depth of the network structure, we show that both factors are relevant in the accuracy of pedaling style prediction. Currently, our best CNN-based Auto-SusPed model predicts a specific composer’s pedaling style and a specific musical era’s style with accuracies of around 90%.

# Directory

First download Maestro V3.0.0. (except .wav files)<br />
Then rebatch files as below.<br />
<br />
composer2composer.py<br />
constants.py<br />
dataset.py<br />
evaluate.py<br />
model.py<br />
train.py<br />
data_analysis<br />
&emspㄴdata_analysis.py<br />
&emspㄴmaestro-v.3.0.0-midi<br />
  &emsp&emspㄴmaestro-v.3.0.0<br />
    &emsp&emspㄴ2004<br />
    &emsp&emsp...<br />
    &emsp&emspㄴ2018<br />
    &emsp&emspㄴcomposers_by_era.xlsx<br />
    &emsp&emspㄴmaestro-v.3.0.0.csv<br />
    &emsp&emspㄴmaestro-v.3.0.0.json<br />
    <br />
Sorry for inconvinience! After I get used to Github, I will make directory more user-friendly!
