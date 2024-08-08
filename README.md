# ADT<sup>2</sup>R: Adaptive Decision Transformer for Dynamic Treatment Regimes in Sepsis

This repository provides the official PyTorch implementation of "Jeon et al., ADT<sup>2</sup>R: Adaptive Decision Transformer for Dynamic Treatment Regimes in Sepsis, IEEE Transactions on Neural Networks and Learning Systems (Accept)."

* Contact: E.-J. Jeon (eunjinjeon@korea.ac.kr)
* We propose an Adaptive Decision Transformer for DTR (ADT$^2$R), which recommends an optimal treatment action for each time step depending on the heterogeneity of the sepsis and a patient's evolving health states. Specifically, we devise a trajectory-optimization-based module to be trained with supervision for treatments and adaptively aggregate the multi-head self-attentions by deliberating on various inherent time-varying patterns among sepsis patients. Furthermore, we estimate the patient's health state by adopting an actor-critic algorithm and inform the treatment recommendation learning about its short-term changes.

## Dataset & Preprocessing
* [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/)
    * Setup MIMIC-III: https://github.com/MIT-LCP/mimic-code 
    * Sepsis preprocessing: https://github.com/uribyul/py_ai_clinician
    * Mechanical ventilation processing: https://github.com/yinchangchang/DAC/tree/main

## Prerequisites
* Python 3.9
* PyTorch 2.0






