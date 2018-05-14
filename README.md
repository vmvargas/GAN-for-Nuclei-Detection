# Semi-Supervised Generative Adversarial Network (SGAN) for Nuclei Detection on Breast Cancer Histopathology Images

This work refine the work of a Semi-Supervised Generative Adversarial Network (SGAN) to work with the H&E breast cancer histopathology images dataset published by the Case Western Reserve University.

Our goal was to evaluate if the semi-supervised model could achieve comparable or better performance to state-of-the-art approaches in nuclei detection. Our results showed that our model was able to learn useful high-level features of nuclear structures as well as to generate visually appealing samples. 

We conclude that more experimentation is necessary in order to formulate a more robust and significant progression.

![G output samples](https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/G-output_samples.png)

## Getting Started

```bash
git clone https://github.com/vmvargas/GAN-for-Nuclei-Detection.git
cd GAN-for-Nuclei-Detection
pip install -r requirements.txt
```

## Dataset

Nuclei Detection in Breast Cancer | Center for Computational Imaging and Personalized Diagnostics. (2018). Engineering.case.edu. Retrieved 10 May 2018, from http://engineering.case.edu/centers/ccipd/data

## Deployment

`model-TMI.py`:  SGAN trained with the TMI Dataset

`model-MNIST.py`:  SGAN trained with the MNIST Dataset

## Built With

- Python 3.6
- Tensorflow 1.6
- Keras 2.1.5