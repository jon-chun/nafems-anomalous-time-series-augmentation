# NAFEMS Conference

## Artificial Intelligence and Machine Learning for Manufacturing
* https://www.nafems.org/events/nafems/2023/artificial-intelligence-and-machine-learning-for-manufacturing/
* 20-21 September 2023

### Multilayered Large Language Models Strategies for Generating Time Series Simulation Data
* https://www.nafems.org/events/nafems/2023/artificial-intelligence-and-machine-learning-for-manufacturing/kenyon-abstract/
* Jon Chun
* Notes and references for NAFEMS talk on augmenting anomalous time series datasets

NAFEMS Anomalous Time Series Augmentation
20 Sep 2023
Jon Chun

Table of Contents:

I. Abstract

II. Introduction and Motivation
   - Problem statement 
   - Research goals

III. Domain Knowledge
   A. Bearing Fault Detection
      - Vibration analysis
      - Frequency transforms
      - Common fault types
   B. Predictive Maintenance
      - Condition monitoring
      - Remaining useful life estimation
   
IV. Literature Review
   A. Time Series Data Augmentation
      - Survey papers
      - GANs, VAEs, etc.
   B. Time Series Anomaly Detection 
      - Classification methods
      - Deep learning techniques
   C. Time Series Forecasting
      - Statistical methods
      - Neural network models
   D. Time Series Generation
      - VAEs, GANs, flows
      - Evaluation metrics
      
V. Datasets
   A. Public Data
      - Case Western bearing data
      - Multivariate time series
   B. Proprietary Data
   
VI. Methods
   A. Data Preprocessing
   B. Feature Engineering
   C. Model Development
      - Architectures
      - Loss functions
   D. Evaluation Metrics
      - Quantitative metrics
      
VII. Experiments and Results
   A. Public Benchmark Results
   B. Proprietary Dataset Results
   C. Synthetic Data Results
   
VIII. Conclusion
   A. Summary
   B. Limitations
   C. Future Work
   
IX. References

==========

Here is the outline with hierarchical numbering for sections and subsections:

I. Abstract

Explore generative deep learning approaches to augment anomalous time series datasets like the Case Western Bearing Dataset in preparation to fine-tune LLM (foundational models) for anomaly detection.

II. Introduction and Motivation

A. Problem Statement



B. Investment and Market Growth: Growth of Synthetic Data
(15 Mar 2023) How synthetic data is boosting AI at scale https://venturebeat.com/ai/synthetic-data-to-boost-ai-at-scale/
(12 Jun 2022) Synthetic Data Is About To Transform Artificial Intelligence https://www.forbes.com/sites/robtoews/2022/06/12/synthetic-data-is-about-to-transform-artificial-intelligence/?sh=19c13de37523 
Gartner Generative AI: https://www.gartner.com/en/insights


III. Bearing Fault Detection

1. Bearing Vibration Analysis 
Predictive Maintenance and Vibration Resources: https://github.com/Charlie5DH/PredictiveMaintenance-and-Vibration-Resources 
Vibrational-Analysis https://github.com/topics/vibrational-analysis
Bearing Fault Detection Vibration Analysis: https://ncd.io/blog/bearing-fault-detection-vibration-analysis/#:~:text=Vibration%20analysis%20is%20a%20widely,issues%2C%20such%20as%20bearing%20faults
Animated Introduction to Vibration Analysis: https://www.youtube.com/watch?v=Vj1xmze3GlE 
Envelope Analysis: https://sensemore.io/envelope-analysis/#:~:text=Envelope%20analysis%2C%20sometimes%20referred%20to,frictional%20forces%20produced%20by%20bearings 
(June 2009) Impact of geometrical defects on bearing assemblies with integrated raceways in aeronautical gearboxes - ScienceDirect 
Phased Array Ultrasonic Testing: https://www.youtube.com/watch?v=g4C-zh51FM0
Applied Vibrational Analysis, Analyzing Bearing Vibrations: https://www.youtube.com/watch?v=53tRsvGmhTY 
PeakVue Vibration Analysis, Outer Race Defect: https://www.youtube.com/watch?v=9QtJbQ61Hsw

2. Frequency Transforms
A Survey on Deep Learning based Time Series Analysis with Frequency Transformation (15 Sep 2023) A Survey on Deep Learning based Time Series Analysis with Frequency Transformation (arxiv.org)



IV. Time Series Topics

A. Time Series and Signal Processing
(3 Feb 2023) cure-lab/Awesome-time-series: A comprehensive survey on the time series domains (github.com) A comprehensive survey on the time series papers from 2018-2022 (we will update it in time ASAP!) on the top conferences (NeurIPS, ICML, ICLR, SIGKDD, SIGIR, AAAI, IJCAI, WWW, CIKM, ICDM, WSDM, etc.)
Research Paper (14 Sep 2023) qingsongedu/awesome-AI-for-time-series-papers: A professional list of Papers, Tutorials, and Surveys on AI for Time Series in top AI conferences and journals. (github.com)
Papers, Libraries, Benchmarks (Feb 2023) cure-lab/Awesome-time-series: A comprehensive survey on the time series domains (github.com)
time-series-analysis · GitHub Topics
time-series · GitHub Topics
TS AI papers, tutorials and surveys (14 Sep 2023) qingsongedu/awesome-AI-for-time-series-papers: A professional list of Papers, Tutorials, and Surveys on AI for Time Series in top AI conferences and journals. (github.com) A professionally curated list of papers (with available code), tutorials, and surveys on recent AI for Time Series Analysis (AI4TS), including Time Series, Spatio-Temporal Data, Event Data, Sequence Data, Temporal Point Processes, etc., at the Top AI Conferences and Journals, which is updated ASAP (the earliest time) once the accepted papers are announced in the corresponding top AI conferences/journals. Hope this list would be helpful for researchers and engineers who are interested in AI for Time Series Analysis.
B. Deep Learning Time Series Forecasting
TS Forecasting and DL (12 Sep 2023) DaoSword/Time-Series-Forecasting-and-Deep-Learning: Resources about time series forecasting and deep learning. (github.com) List of research papers focus on time series forecasting and deep learning, as well as other resources like competitions, datasets, courses, blogs, code, etc.
(Aug 2023) OmniXAI salesforce/OmniXAI: OmniXAI: A Library for eXplainable AI (github.com)
(24 Mar 2023) XAI for Forecasting: Basis Expansion | by Nakul Upadhya | Towards Data Science 
C. Time Series Anomaly Detection
TS Anomaly Detection Resources:  (6 Jun 2023) yzhao062/anomaly-detection-resources: Anomaly detection related books, papers, videos, and toolboxes (github.com) 
(21 Sep 2022) rob-med/awesome-TS-anomaly-detection: List of tools & datasets for anomaly detection on time-series data. (github.com) 
(3 Jul 2023) zamanzadeh/ts-anomaly-benchmark: Time-Series Anomaly Detection Comprehensive Benchmark (github.com) This repository updates the comprehensive list of classic and state-of-the-art methods and datasets for Anomaly Decetion in Time-Series. This is part of an onging research at Time Series Analytics Lab, Monash University.
Time Series Anomaly Detection: time-series-anomaly-detection · GitHub Topics
(26 Dec 2021) https://github.com/tvhahn/Manufacturing-Data-Science-with-Python with Colabs
Papers:
(4 Sep 2023) DiffAD ChunjingXiao/DiffAD: Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models (github.com)
(4 Sep 2023) MSTICPY MS Threat Intelligence Python Tools microsoft/msticpy: Microsoft Threat Intelligence Security Tools (github.com)
(8 May 2023) linkedin/luminol: Anomaly Detection and Correlation library (github.com)
(9 May 2022) MTS Deep Learning Anomaly Detection astha-chem/mvts-ano-eval: A repository for code accompanying the manuscript 'An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series' (published at TNNLS) (github.com)
(31 May 2020) ML and LSTM Outlier Detection DawidSitnik/Anomaly-Detection-in-Time-Series-Datasets: This is project made for one of the subjects at Warsaw University of Technology. Its aim is to detect anomaly in time series data. (github.com)
D. Time Series Synthesis
1. Statistical ML
GRATIS.R robjhyndman.com https://www.youtube.com/watch?v=F3lWECtFa44 
ADSN Conf Syn Data: https://www.youtube.com/watch?v=F3lWECtFa44
Tabular/Statistical: sdv-dev/SDV: Synthetic data generation for tabular data (github.com)
CTGAN: ydataai/ydata-synthetic: Synthetic data generators for tabular and time-series data (github.com) 
uchidalab/time_series_augmentation: An example of time series augmentation methods with Keras (github.com) 
2. Deep Learning
Kera Tutorials: https://keras.io/examples/generative/
Data Augmentation techniques in time series domain: a survey and taxonomy (24 Mar 2023) https://link.springer.com/article/10.1007/s00521-023-08459-3 
TTransFusion: Generating Long, High Fidelity Time Series using Diffusion Models with Transformers (24 Jul 2023) 



V. Vibrational Datasets and Code

A. Public Data

1. Case Western Bearing Data
Case Western Bearing (5 Dec 2021): Jpickard1/BallBearings: This repository contains data and code to recreate classification results for fault detection in ball bearings. The data comes from the Case Western Reserve Bearing Data Center (github.com)
Github (17 Feb 2021): s-whynot/CWRU-dataset: Case Western Reserve University Bearing Fault Dataset (github.com) 
Github Metadata for Python (14 Apr 2020): ryanjung94/cwru_py3: Collect Case Western Reserve University Bearing Data in python 3 (github.com) 
Github (14 May 2022): Data-of-Case-Western-Reserve-University/Normal Baseline Data at main · yzbbj/Data-of-Case-Western-Reserve-University (github.com)

2. Multivariate Time Series
M4 Dataset and Competition: https://github.com/Mcompetitions/M4-methods 
MTS laiguokun/multivariate-time-series-data (github.com)
Anomaly Datasets (10 Sep 2023) GuansongPang/ADRepository-Anomaly-detection-datasets: ADRepository: Real-world anomaly detection datasets, including tabular data (categorical and numerical data), time series data, graph data, image data, and video data. (github.com)
AI4I 2020 UCI PdM  AI4I 2020 Predictive Maintenance Dataset - UCI Machine Learning Repository 
3. Bearing Vibrational Code
Rolling element bearing fault diagnosis using convolutional neural network and vibration image (sciencedirectassets.com) (2019) Hoang & Kang
[PDF] Fault Detection in Ball Bearings | Semantic Scholar (19 Sep 2022) Pickard & Moll
Case Western Bearing Colab EDA (5 Dec 2021): Jpickard1/BallBearings: This repository contains data and code to recreate classification results for fault detection in ball bearings. The data comes from the Case Western Reserve Bearing Data Center (github.com)
(27 Nov 2019) raady07/CNN-for-bearing-fault-diagnosis: CNN applied to bearing signals for analysis (github.com) 
SB-PdM: a tool for predictive maintenance of rolling bearings based on limited labeled data (Feb 2023) https://github.com/Western-OC2-Lab/SB-PdM-a-tool-for-predictive-maintenance-of-rolling-bearings-based-on-limited-labeled-data/blob/main/SB_PdM_Tool.ipynb 
Vibration-Based Fault Diagnosis with Low Delay (26 Jan 2023) https://github.com/Western-OC2-Lab/Vibration-Based-Fault-Diagnosis-with-Low-Delay 
Rolling element bearing fault diagnosis using convolutional neural network and vibration image (11 Jul 2021) Github lestercardoz11/fault-detection-for-predictive-maintenance-in-industry-4.0: This research project will illustrate the use of machine learning and deep learning for predictive analysis in industry 4.0. (github.com) Nine colab notebooks for machine learning and deep learning for predictive analysis in industry 4.0.
Github (29 Sep 2022) Yi-Chen-Lin2019/Predictive-maintenance-with-machine-learning: This project is about predictive maintenance with machine learning. It's a final project of my Computer Science AP degree. (github.com) Supervised and unsupervised models for 3 tasks, 1. Anomaly detection, 2. Remaining useful life and 3. Failure prediction on 2 datasets a. CW Bearing and b. NASA Battery
Github *.R (3 Jan 2023) Miltos-90/Failure_Classification_of_Bearings: Failure Mode Classification from the NASA/IMS Bearing Dataset (github.com) 
Github (9 Feb 2022) https://github.com/devamsheth21/Bearing-Fault-Detection-using-Deep-Learning-approach  Detection and multi-class classification of Bearing faults using Image classification from Case Western Reserve University data of bearing vibrations recorded at different frequencies. Developed an algorithm to convert vibrational data into Symmetrized Dot Pattern images based on a Research paper. Created an Image dataset of 50 different parameters and 4 different fault classes, to select optimum parameters for efficient classification. Trained and tested 50 different datasets on different Image-net models to obtain maximum accuracy. Obtained an accuracy of 98% for Binary classification of Inner and Outer race faults on Efficient Net B7 model on just 5 epochs. 
Github (15 Dec 2022) https://github.com/MideTechnology/endaq-python A comprehensive, user-centric Python API for working with enDAQ data and devices Manual: https://docs.endaq.com/en/latest/api_ref.html Video: https://endaq.com/pages/endaq-open-source-python-library-for-shock-vibration-analysis 
Github (15 Nov 2020) https://github.com/Vibration-Testing/vibrationtesting Manual: http://vibration-testing.github.io/vibrationtesting/installation.html 
Github (30 Sep 2021) TadGAN (w/LSTM) arunppsg/TadGAN: Code for the paper "TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks" (github.com) Paper 2009.07769.pdf (arxiv.org)
(25 Jan 2023) Survey of Methods Dhueper/TimeSeries-AnomalyDetection: Anomaly detection in time series for space applications (github.com)
(25 Jan 2022) PimW/Time-Series-Anomaly-Detection: Time series analysis using a grammar based compression algorithm. (github.com) Uses the discretization used for time series in PySAX and the grammar based compression of Sequitur as basis for the compression of the time series. The algorithm then uses the compression to calculate a score of the compressibility of each point in the time-series. If the compressibility of a sequence of points is low for a certain sequence then an anomaly is detected.

VI. Model Architecture Types (some mixed)

A. RNN: LSTM/GRU
Visualization: LSTMViz (19 Nov 2021) https://github.com/HendrikStrobelt/lstmvis 
Paper: LSTM and GRU Neural Networks as Models of Dynamical Processes Used in Predictive Control: A Comparison of Models Developed for Two Chemical Reactors (17 Aug 2021) https://www.mdpi.com/1424-8220/21/16/5625
Model: AE-RNN https://github.com/RobRomijnders/AE_ts 
B. VAE
Overview: https://riccardo-cantini.netlify.app/post/cnn_vae_mnist/
VAE Keras Tutorial: https://keras.io/examples/generative/vae/ 
VAE Time Series Anomaly Detection UC Berkeley Milling Dataset:
Tutorial: https://www.tvhahn.com/posts/building-vae/
Colab: https://github.com/tvhahn/Manufacturing-Data-Science-with-Python/tree/master/Metal%20Machining
C. GAN
1. Open Source: DoppleGANger 
Paper (17 Jan 2021) Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions (arxiv.org)
Repo (12 Aug 2023) fjxmlzn/DoppelGANger: [IMC 2020 (Best Paper Finalist)] Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions (github.com)
2. Commercial: Gretel.ai DGAN
Overview: https://www.youtube.com/watch?v=XVPFR-P6vlg&t=279s
DGAN Doc: https://docs.gretel.ai/reference/synthetics/models/gretel-dgan
DGAN Workshop (5 Oct 2022): https://www.youtube.com/watch?v=YF3ivHw9KmA 
DGAN Blog (15 Sep 2022): https://gretel.ai/blog/generate-time-series-data-with-gretels-new-dgan-model  
Colab (May 2023) https://github.com/gretelai/gretel-blueprints/blob/main/docs/notebooks/create_synthetic_data_from_dgan_api.ipynb 
DoppleGANger Pytorch (21 Jun 2022): https://gretel.ai/blog/create-synthetic-time-series-with-doppelganger-and-pytorch 
	3. Commercial: yData
Why synthetic data (ydata.ai)  
	4. WGAN Tutorial:
https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
WGAN/1-Lipschitz constraint: https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490 
	5. CGAN
Keras Code: Conditional GAN (CGAN): https://keras.io/examples/generative/conditional_gan/ 
	6. Newer GAN Models
TimeGAN
(14 Oct 2022) jsyoon0823/TimeGAN: Codebase for Time-series Generative Adversarial Networks (TimeGAN) - NeurIPS 2019 (github.com)
TimeSynth
(20201130) TimeSynth/TimeSynth: A Multipurpose Library for Synthetic Time Series Generation in Python (github.com) TimeSynth is an open source library for generating synthetic time series for model testing. The library can generate regular and irregular time series. The architecture allows the user to match different signals with different architectures allowing a vast array of signals to be generated.
LTSNet 
(27 Apr 2022) fbadine/LSTNet: A Tensorflow / Keras implementation of "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" paper (github.com)
(21 Dec 2019) laiguokun/LSTNet (github.com) 
RTSGAN
acphile/RTSGAN (github.com)
TSGAN for Cloud Workload
(16 Dec 2022) soraminnnn/TsGAN: PyTorch implementation of A GAN-based method for time-dependent cloud workload generation. (github.com)
TSGAN for Biology
(13 Dec 2019) numancelik34/TimeSeries-GAN: Generation of Time Series data using generatuve adversarial networks (GANs) for biological purposes. (github.com)

D. Transformers/Attention Heads
Paper: Transformers in Time Series: A Survey (11 May 2023) 2202.07125.pdf (arxiv.org) 

E. Energy


F. Diffusion
Huggingface Diffusers: Diffusers (huggingface.co)
Video: https://www.youtube.com/watch?v=fbLgFrlTnGU
V. Multi-Model Frameworks / Benchmarks
Synthcity
Github (20230912 208 stars) vanderschaarlab/synthcity: A library for generating and evaluating synthetic tabular data for privacy, fairness and data augmentation. (github.com)	
TSGM (VAEs, GANS, Metrics)
Github (20230918 31 stars) Alexander Nikitin/tsgm: Generative modeling of synthetic time series data and time series augmentations (github.com)
Documentation (pypi) Time Series Simulator (TSGM) Official Documentation — tsgm 0.0.0 documentation
Colab: Getting started with TSGM.ipynb - Colaboratory (google.com)  
SB-PdM (Similarity-Based Predictive Maintenance) Feat Ext/Sim Metrics
Repo: SB-PdM-a-tool-for-predictive-maintenance-of-rolling-bearings-based-on-limited-labeled-data/SB_PdM_Tool.ipynb at main · Western-OC2-Lab/SB-PdM-a-tool-for-predictive-maintenance-of-rolling-bearings-based-on-limited-labeled-data (github.com)
Colab: SB-PdM-a-tool-for-predictive-maintenance-of-rolling-bearings-based-on-limited-labeled-data/SB_PdM_Tool.ipynb at main · Western-OC2-Lab/SB-PdM-a-tool-for-predictive-maintenance-of-rolling-bearings-based-on-limited-labeled-data (github.com)
Flow-Forecast
(12 Sep 2023) https://github.com/AIStream-Peelout/flow-forecast Flow Forecast (FF) is an open-source deep learning for time series forecasting framework. It provides all the latest state of the art models (transformers, attention models, GRUs, ODEs) and cutting edge concepts with easy to understand interpretability metrics, cloud provider integration, and model serving capabilities. Flow Forecast was the first time series framework to feature support for transformer based models and remains the only true end-to-end deep learning for time series framework.  
Manual: https://flow-forecast.atlassian.net/wiki/spaces/FF/pages/92864513/Getting+Started 
Tutorials: https://github.com/AIStream-Peelout/flow_tutorials 

VI. SOTA Research
C-GATS
Paper: c-gats-conditional-generation-of-anomalous-time-series.pdf (amazon.science)
OpenReview.org (5 May 2023) C-GATS: Conditional Generation of Anomalous Time Series | OpenReview 
IH-TCGAN (1 May 2023)
Paper: Entropy | Free Full-Text | IH-TCGAN: Time-Series Conditional Generative Adversarial Network with Improved Hausdorff Distance for Synthesizing Intention Recognition Data (mdpi.com) 
ImDiffusion
17000cyh/IMDiffusion (github.com)
TransFusion
https://arxiv.org/pdf/2307.12667.pdf 

VII. Evaluation Metrics

1. Leaderboards
M4: https://paperswithcode.com/dataset/m4
MVTec AD: MVTec AD Benchmark (Anomaly Detection) | Papers With Code 

2. Testing Frameworks and Benchmarks
(24 Feb 2022) numenta/NAB: The Numenta Anomaly Benchmark (github.com) Numenta Anomaly Benchmark (NAB) v1.1 is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications. It is composed of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications.
(6 Aug 2023) DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models (github.com) Summarize the evaluation metrics used in unconditional generative models for synthetic data generation, list the advantages and disadvantages of each evaluation metric based on experiments on different datasets and models.  We implement some popular models for time series generation including: Time-GAN, Recurrent Conditional GAN (RCGAN), Time-VAE. 
TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series (arxiv.org)
AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data (7 Jun 2023) emadeldeen24/AdaTime: [TKDD 2023] AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data (github.com)


VIII. Future Directions
Graph Neural Networks 
(25 Dec 2021) [2010.05234] A Practical Tutorial on Graph Neural Networks (arxiv.org)
https://arxiv.org/pdf/2306.11768.pdf 
Hands-On Graph Neural Networks Using Python: Practical techniques and architectures for building powerful graph and deep learning apps with PyTorch: Labonne, Maxime: 9781804617526: Amazon.com: Books 
Geometric Deep Learning
Towards Geometric Deep Learning (thegradient.pub) 
[2306.11768] A Systematic Survey in Geometric Deep Learning for Structure-based Drug Design (arxiv.org)
(13 Jul 2023) Frontiers | Geometric deep learning as a potential tool for antimicrobial peptide prediction (frontiersin.org)
Autonomous Agents
Multimodal AnamolyGPT (3 Sep 2023) CASIA-IVA-Lab/AnomalyGPT: The first LVLM based IAD method! (github.com)
More Through Reasoning: GoT: spcl/graph-of-thoughts: Official Implementation of "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (github.com)
(22 Aug 2023) A Survey on Large Language Model based Autonomous Agents
Paper: https://arxiv.org/abs/2308.11432
Repo: Paitesanshi/LLM-Agent-Survey (github.com)
Live Leaderboard: LLM-based Autonomous Agent (notion.site)
List of Research Papers: https://github.com/WooooDyy/LLM-Agent-Paper-List

IX. Useful Texts
Generative Deep Learning, 2nd Ed. by David Foster (O’Reilly, June 2023)
Book: https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1098134184/ref=sr_1_1?keywords=generative+deep+learning&qid=1694960894&sr=8-1 
Code Repo: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition 
Probabilistic Machine Learning: An Introduction by Kevin Murphy
https://probml.github.io/pml-book/book1.html 





