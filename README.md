# AutoML4EO
Repository for the Master thesis project "Automated Machine Learning for Satellite Data: Integrating Remote Sensing Pre-trained Models into AutoML Systems"

Current AutoML systems have been benchmarked with natural image datasets. However, there are various differences between satellite images and natural images, for instance in the bit-wise resolution, the number and type of spectral bands, which open questions about the applicability of current AutoML systems for satellite data tasks. In this thesis, we demonstrate how AutoML can be leveraged for classification tasks on satellite data. Specifically, we deploy the Auto-Keras system for image classification tasks and create two new variants of it for satellite image classification that incorporate transfer learning using models pre-trained with (i) natural images (using ImageNet) and (ii) remote sensing datasets.
For evaluation, we compared the performance of these variants against manually designed architectures on a benchmark set of 7 satellite datasets.

### Datasets ###

  * [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet "BigEarthNet"): G. Sumbul, M. Charfuelan, B. Demir, V. Markl. "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding", IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.
  * [EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat "EuroSAT"):  P. Helber, B. Bischke, A. Dengel, D. Borth. "Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
  * [So2Sat Sentinel-2](https://www.tensorflow.org/datasets/catalog/so2sat "So2Sat"): X. Zhu, et al. "So2sat lcz42", 2018. https://doi.org/10.14459/2018MP1454690
  * [UC Merced](https://www.tensorflow.org/datasets/catalog/uc_merced "UCMerced"):  Y. Yang, S. Newsam. "Bag-of-visual-words and spatial extensions for land-useclassification", ACM SIGSPATIAL International Conference on Advances inGeographic Information Systems (ACM GIS), 2010.
  * [BrazilDam Sentinel-2 (2019)](http://www.patreo.dcc.ufmg.br/2020/01/27/brazildam-dataset/ "BrazilDAM"): E. Ferreira, M. Brito, M. S. Alvim,R. Balaniuk, J. dos Santos. "BrazilDAM: A Benchmark dataset for Tailings Dam Detection", Latin American GRSS & ISPRS Remote Sensing Conference 2020, Santigo, Chile, 2020.
  * [Brazilian Cerrado-Savanna Scenes](http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-cerrado-savanna-scenes-dataset/ "CerradoSavana"): K. Nogueira, J. A. dos Santos, T. Fornazari, T. S. Freire, L. P. Morellato, R. da S. Torres. "Towards Vegetation Species Discrimination by using Data-driven Descriptors", International Conference on Pattern Recognition Workshop in Pattern Recogniton in Remote Sensing, Cancun, Mexico, 2016.
  * [Brazilian Coffee Scenes](http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-coffee-scenes-dataset/ "BrazilianCoffee"): O. A. B. Penatti, K. Nogueira, J. A. dos Santos. "Do Deep Features Generalize from Everyday Objects to Remote Sensing and Aerial Scenes Domains?", IEEE Computer Vision and Pattern Recognition Workshops In EarthVision 2015, Boston, 2015.

### Remote sensing pre-trained models ###
  * The 3-channel pre-trained models are available at https://tfhub.dev/google/collections/remote_sensing/1 (M. Neumann, A. Susano Pinto, X. Zhai, N. Houlsby "Training general representations for remote sensing usingin-domain knowledge", IEEE International Geoscience and Remote SensingSymposium, 2020.)
  * The 13-channel pre-trained models can be downloaded from https://www.dropbox.com/sh/l7j9591teap0xvm/AAAvTffeTZV2j9wXP3rWzhgea?dl=0
  
### Contents ###
The structure of this repository is the following:

-autokeras: Auto-Keras variant with a new remote sensing block and a task specific 'SatelliteImageClassifier'.
Modified version of [AutoKeras](https://github.com/keras-team/autokeras "AutoKeras") (H. Jin, Q. Song, and X. Hu. "Auto-keras: An efficient neural architecture search system." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019.)

-datasets: Code to build the datasets and scripts to add the missing data sources to 'tensorflow-datasets'

-experiments

-results


#### Basic example of using the satellite image classification API. ####
```
import autokeras as ak
clf = ak.SatelliteImageClassifier()
clf.fit(X1,y1)
print(clf.evaluate(X2,y2))
results =clf.predict(X2)
```
### Cite this work ###
Nelly R. Palacios Salinas (supervisors: Dr. Mitra Baratchi & Dr. Jan van Rijn & Dr. Andreas Vollrath)
Automated Machine Learning for Satellite Data: Integrating remote sensing pre-trained models into AutoML systems.
Master's Thesis in Computer Science at Leiden Institute of Advanced Computer Science, Leiden University, 2021. ([Download](https://ada.liacs.nl/papers/PalEtAl21b.pdf))

Nelly Rosaura Palacios Salinas, Mitra Baratchi, Jan N. van Rijn & Andreas Vollrath
Automated Machine Learning for Satellite Data: Integrating remote sensing pre-trained models into AutoML systems.
In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. 2021, to appear. ([Download](https://ada.liacs.nl/papers/PalEtAl21.pdf))

