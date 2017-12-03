# Cat GAN

This is the GAN implementation of cats generator. 

What is the GAN? Generative adversarial networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework ([link](https://en.wikipedia.org/wiki/Generative_adversarial_network)). So, it means that we should train 2 networks with different architectures and in different ways.

## Architecture

### Generator

The generator has very custom but straightforward architecure.
ConvTranspose2d -> SELU -> [ConvTranspose2d -> SELU] x N -> ConvTranspose2d -> Tanh

### Discriminator

A ResNet-18 with single-neuron output seems compatible for this project.

The generator has very custom but straightforward architecure.
ConvTranspose2d -> SELU -> [ConvTranspose2d -> SELU] x N -> ConvTranspose2d -> Tanh

## Examples

The output on 5k iteration (for default configuration):
![2017-11-15 21-42-34](https://user-images.githubusercontent.com/3521007/32853968-22e3659c-ca4e-11e7-9eeb-c04663f33388.png)
5 minute after:
![2017-11-15 21-47-37](https://user-images.githubusercontent.com/3521007/32854134-b179687e-ca4e-11e7-81cd-eaf52ecc71fb.png)
2 hours after:
![2017-11-16 00-07-49](https://user-images.githubusercontent.com/3521007/32860436-a795ea62-ca62-11e7-979f-a65512605dbe.png)

## Installation
```
pip install -r requirements.txt
```
For the PyTorch installation, please follow [this guide](http://pytorch.org).

## Training
To run the training script, make changes in the configuration (example is in config/train.yml).

```
python train.py --config config/train.yml
```

## Testing
To get radom cat (fill restore.generator in config/train.yml):

```
python test.py --config config/train.yml
```

![2017-11-18 01-18-30](https://user-images.githubusercontent.com/3521007/32971483-76116488-cbfe-11e7-8271-4c85241d573d.png)
![2017-11-18 01-18-49](https://user-images.githubusercontent.com/3521007/32971485-76475908-cbfe-11e7-84ff-68bae185d8c9.png)


## References
Inspired by https://arxiv.org/abs/1406.2661

## License
This software is covered by MIT License.
