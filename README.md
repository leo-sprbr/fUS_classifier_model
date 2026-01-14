### Project overview

This project aims to train a video classifier model in order to classify macaque behavior according to functional ultrasound imaging during an oculomotor task. 
This is a project I have been working on for the past 4 months within the MOV'IT lab at the Paris Brain Institute under the supervision of Dr. Pierre Pouget.

Functional Ultrasound Imaging (fUS) is a brain imaging technique that provides very high spatiotemporal resolution for recording hemodynamic activity. It works by capturing planar ultrasound echoes at a high frequency, and through singular value decomposition (SVD), it generates images that highlight the Doppler signal from red blood cells, providing a proxy measure of cerebral blood flow. In our case, fUS was taken in macaques during an oculomotor saccade-antisaccade task, during which the monkeys sometimes decided to stop doing the tasking, taking a "pause". This has been researched and analysed in a [previous study](https://doi.org/10.1101/2025.04.01.646546) where a CNN classifier was built and was shown to be able to classify the monkey's behavior (if it was in a state of "work" or "pause") only based on single fUS frames.
However, this model only work within each acquisition, meaning it trained and then tested itself on the frames of one single recording session. The next logical step was then to ask weather it would be possible to 
monitor the monkey's behavior in an online fashion, even imagining closed-loop applications of such a model. It's with this idea that this project, and my internship, started this September of 2025.

The base concept is to have a classifier model that is able to classify the monkey's behavior based on fUS recordings that it has never seen before. So instead of training and then classifying within a single session,
the model should be able to train on a set of recordings, and then classify an entirely new set of recordings. Also, the model cannot use "future" information to help itself, as in an online paradigm, it'll only have 
the current, and previous fUS recording to work with. Meaning for example that in the [Julien Claron et al.](https://plexon.com/software-downloads/#software-downloads-SDKs) paper, the polynomial fit of the session that was subtracted from the cerebral blood volume in order to combat
cognitive fatigue, will not be possible. With these limitations in mind, and with the scope of pushing as far as possible the accuracy of this model, I decided to train a video classifier, instead of an image classifier.
The fUS dataset is divided into frame "patches" of 8, 16 or 32 consecutive frames, which are labeled with the last frame's label (0 for work and 1 for pause). In an online paradigm, this would mean the model would be
classifying each new frame using the frames coming before it. The rationale behind this decision was linked to the nature of the recording. With this kind of spatiotemporal resolution, the recording of hemodynamic
response can show very different curve shapes, phase and propagation through the region, all information that are lost with single frames, whereas, using "patches" of 16 frames, which at 2.5 Hz represent a 6.4s video,
making these dynamics clearly visible. The hypothesis is that they will help the model better classify the monkey's behavior. 

*"If I had 10 hours to work on a deep learning model, I'd spend the first 6 on curating my dataset"* -some smart person

For anyone taking back the project, there is **one very important thing to consider**. One of the reason I have not yet been able to produce satisfying results with this model is the recent realisation that the dataset
I was working with included recording sessions with different experimental designs, such as variations in the inter-trial time. This meant periods in the acquisition where the monkey was completing one trial every 10 seconds
instead of every 3 seconds. The model then recognised these moments in the hemodynamic data as "pauses", because it was indeed a decrease in activity, without them being labeled as pause, making the training dataset erroneous.
All that to say, that the model should, in theory, be able to classify these different cognitive states, but that it is absolutely crucial, for multi-session classifying, that the sessions are experimentally consistent
and that the labels actually reflect the monkey's behavior.

### Code structure
We will now go over the different files and their purpose:

**processing_and_eval.ipynb** : main code of this project. It is a python notebook containing all the loading, pre-processing, masking, dataset creation and model evaluation code. The main thing to consider
is the format of the image and label data. Everything here was done with processed power doppler images, which will have to be done if working on raw IQ. Also, the label were already given for the dataset, if absent 
refer to [Julien Claron et al.](https://doi.org/10.1101/2025.04.01.646546) for the pause labeling.

**extract_plx.m** : matlab file to extract the behavioral plexon data, including eye movement, trials, reward outcome, fUS frame timestamps and pupil size. In order to use it you will have to install the 
[matlab offline plexon SDK](https://plexon.com/software-downloads/#software-downloads-SDKs) locally and add its path to the file 

**behavioral_analysis.ipynb** : python notebook to analyse the data extracted with extract_plt.m. Creates a dataframe containing each trial with its nature, outcome and event timestamp. Since we are classifying behavioral
data, sometimes the model's classification can be explained by looking at the corresponding behavior of the monkey. 

**helper_functions.py** : contains all the processing functions used in the notebooks.

**trainMAEsmallbalance.py** and **trainMAEsmallweighted.py** : the actual training of the model was not done locally but on the institute's GPU cluster. These are the script that were used for this training, using
a 50/50 balanced dataset and a weighted loss function respectively. 

### Next steps

Here are a few suggestions on the next steps towards creating an online classifier model. 

1. **Curating a consistent dataset**. Again, not having the behavioral information led me to train the model on very inconsistent data, which meant it could hardly learn anything. The most important thing here is to be sure about your data and its labels. Be careful about the experimental design and the images itself (I've had a few horizontally flipped acquisitions). The best way for the model to learn, is for each acquisition to be as similar to one another as possible.
2. **Pre-training the model on fUS data**. The model we're finetuning here is pre-trained on [kinetics](https://github.com/cvdfoundation/kinetics-dataset), therefore it is more used to classify human movement than noisy, textured and static fUS data. Pre-training the model to another fUS dataset or even ultrasound imaging, could probably boost its performance, as its been seen in [other studies](https://doi.org/10.1101/2024.10.09.24315195)
3. **Changing the labelling method**. In [Julien Claron et al.](https://doi.org/10.1101/2025.04.01.646546) the pause label is defined as 3 consecutive trials not completed by the monkey. This doesn't consider the difference there can be between periods where the monkey does every trial and periods where the monkey skips one trial every four. It could be a good idea, instead of implementing a binary classification between "work" and "pause", we could create a sort of "gradient of attention" of the monkey during the task.
4. **Building the online paradigm**. The true purpose of this model would be to be used in an online setting. If the model shows satisfactory results, then we should code a simulation of online processing. Feeding the program fUS frames at the speed they will be taken and finetune it to be capable to process and classify each frame as it comes. 

For any additional questions, feel free to contact leo.sperber@gmail.com

Useful refs:

- Mishra, D., Salehi, M., Saha, P., Patey, O., Papageorghiou, A. T., Asano, Y. M., & Noble, J. A. (2025). Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation. arXiv.org. https://doi.org/10.48550/arxiv.2506.11777

- Howard, J. P., Tan, J., Shun-Shin, M. J., Mahdi, D., Nowbar, A. N., Arnold, A. D., … Francis, D. P. (2020). Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography. Journal of Medical Artificial Intelligence. https://doi.org/10.21037/jmai.2019.10.03

- Yue, Y., & Li, Z. (2024). MedMamba: Vision Mamba for Medical Image Classification. arXiv.org. https://doi.org/10.48550/arxiv.2403.03849

- Kang, Q., Lao, Q., Gao, J., Bao, W., Zhu, H., Du, C., Qiang, L., & Li, K. (2025). URFM: a general Ultrasound Representation Foundation Model for advancing ultrasound image diagnosis. iScience. https://doi.org/10.1016/j.isci.2025.112917

- Claron, J., Blons, M., Dizeux, A., Deffieux, T., Tanter, M., Berthon, B., & Pouget, P. (2025). Distributed Activity in the Medial Frontal Cortex Predicts Self-Initiated Action. bioRxiv. https://doi.org/10.1101/2025.04.01.646546

- Di Ianni, T., & Airan, R. (2022). Deep-fUS: A Deep Learning Platform for Functional Ultrasound Imaging of the Brain Using Sparse Data. IEEE Transactions on Medical Imaging. https://doi.org/10.1109/tmi.2022.3148728

- Deighton, J., Zhong, S., Agyeman, K., Choi, W., Liu, C. Y., Lee, D., Maroulas, V., & Christopoulos, V. (2025). Functional Ultrasound Imaging Combined with Machine Learning for Whole-Brain Analysis of Drug-Induced Hemodynamic Changes. Imaging Neuroscience. https://doi.org/10.1162/imag.a.139

- Lambert, T., Niknejad, H., Kil, D., Montaldo, G., Nuttin, B., Brunner, C., & Urban, A. (2025). Spatiotemporal Clustering of Functional Ultrasound Signals at the Single-Voxel Level. eNeuro. https://doi.org/10.1523/eneuro.0438-24.2025

- Zhang, Z., Wu, Q., Ding, S., Wang, X., Ye, J., & San Francisco. (2024). EchoFM: A Pre-training and Fine-tuning Framework for Echocardiogram Videos Vision Foundation Model. medRxiv. https://doi.org/10.1101/2024.10.09.24315195

- Griggs, W. S., Norman, S., Deffieux, T., Segura, F., Osmanski, B., Chau, G., Christopoulos, V., Liu, C., Tanter, M., Shapiro, M. G., & Andersen, R. A. (2023). Decoding motor plans using a closed-loop ultrasonic brain–machine interface. bioRxiv. https://doi.org/10.1038/s41593-023-01500-7

- Liu, B., Luo, Q., Liang, Z., He, H., & Gu, Y. (2025). Robust single-trial decoding of physical self-motion from hemodynamic signals in the brain measured by functional ultrasound imaging. PNAS. https://doi.org/10.1073/pnas.2414354122

- Griggs, W. S., Norman, S. L., Tanter, M., Liu, C. Y., Christopoulos, V., Shapiro, M. G., & Andersen, R. A. (2025). Functional ultrasound neuroimaging reveals mesoscopic organization of saccades in the lateral intraparietal area. Nature Communications. https://doi.org/10.1038/s41467-025-63826-z

- Norman, S., Maresca, D., Christopoulos, V., Griggs, W. S., Demené, C., Tanter, M., Shapiro, M. G., & Andersen, R. A. (2020). Single-trial decoding of movement intentions using functional ultrasound neuroimaging. Neuron. https://doi.org/10.1101/2020.05.12.086132

- Wu, C. H., Tsai, C. J., & Kuo, P. C. (2025). From Visualization to Automation: A Narrative Review of Deep Learning’s Impact on Ultrasound-based Median Nerve Assessment. Journal of Medical Ultrasound. https://doi.org/10.4103/jmu.jmu-d-25-00010

- Cui, X., Li, Z., Fan, X., Huang, P., Wang, Y., Yang, M., Chang, S., & Zhu, J. (2025). Variable-frame CNNLSTM for Breast Nodule Classification using Ultrasound Videos. arXiv.org. https://doi.org/10.48550/arxiv.2502.11481

Author: Leo Sperber

