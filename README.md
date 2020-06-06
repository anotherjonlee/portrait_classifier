# portrait_classifier
The project seeks to identify Roman emperors from Roman imperial coin features.
# Table of Contents

- [Introduction](#introduction)
- [EDA](#eda)
- [Basic Model](#basic-model)
- [Transfer Learning](#transfer-learning)
- [Conclusion](#conclusion)
- [Future Plans](#future-plans)
- [Sources](#sources)
# Introduction

# EDA
* Unbalanced data
![unbalanced_data](data/images/emperors_coins.png)
* EDA after balancing data
![history](data/images/denom_hist.png)

# Data manipulation
<p float="left">
  <img src="data/images/original.png" width="200" />
  <img src="data/images/manipulated_coins_0_9765.png" width="200"/>
  <img src="data/images/manipulated_coins_0_1744.png" width="200"/> 
</p>

# Basic Model

## Model architecture 

## Results

<p float="left">
  <img src="data/images/model_acc.png" width="400" />
  <img src="data/images/model_loss.png" width="400" /> 
</p>

# Transfer Learning Model (VGG-16)

## Model architecture

## Result
<p float="left">
  <img src="data/images/final_vgg_accuracy.png" width="400" />
  <img src="data/images/final_vgg_loss.png" width="400" /> 
</p>

# Conclusion

![precisionrecall](data/images/precision_recall.png)

<p float="left">
  <img src="data/images/gallienus.png" width="200" />
  <img src="data/images/trajan.png" width="200" /> 
  <img src="data/images/galerius.png" width="214" />
</p>

![confusion](data/images/cm.png)

<p float="left">
  <img src="data/images/constantine1.png" width="206" />
  <img src="data/images/constantine2.png" width="200" /> 
  <img src="data/images/constantius2.png" width="224" />
</p>



# Future Plans
- Gray scale images to negate material differences, since the model picked up correlation among coins minted under the reign of Constantine I, Constantine II and Constantius II. 
- Use unbalanced data

# Sources
- http://numismatics.org
- https://core.ac.uk/download/pdf/110425364.pdf
