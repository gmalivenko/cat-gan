# Cat GAN

This is the GAN implementation of cats generator.

The output on 5k iteration (for default configuration):
![2017-11-15 21-42-34](https://user-images.githubusercontent.com/3521007/32853968-22e3659c-ca4e-11e7-9eeb-c04663f33388.png)
5 minute after:
![2017-11-15 21-47-37](https://user-images.githubusercontent.com/3521007/32854134-b179687e-ca4e-11e7-81cd-eaf52ecc71fb.png)
2 hours after:
![2017-11-16 00-07-49](https://user-images.githubusercontent.com/3521007/32860436-a795ea62-ca62-11e7-979f-a65512605dbe.png)

# Installation
```
pip install -r requirements.txt
```
For the PyTorch installation, please follow [this guide](http://pytorch.org)

# Training
To run the training script, make changes in the configuration (example is config/train.yml).

```
python train.py --config config/train.yml
```

# References
Inspired by https://arxiv.org/abs/1406.2661

# License
This software is covered by MIT License.
