# deep-learning-papers

## Computer Vision
### CNN Architecture
* AlexNet: ImageNet Classification with Deep Convolutional Neural Networks ([note](https://drive.google.com/open?id=1oxzo2ZulvusLn_ETVK5mTPlgqPJLs9HA))
* ZFNet (DeconvNet): Visualizing and Understanding Convolutional Networks ([note](https://drive.google.com/open?id=1bzkoKVxLALaD6ZWQh5-vP9qOodMdIwi0), code)
* NIN: Network in Network
* VggNet: Very Deep Convolutional Networks for Large-Scale Image Recognition
* GoogLeNet: Going Deeper with Convolutions
* ResNet:
  - ResNet-v1: Deep Residual Learning for Image Recognition ([note](https://drive.google.com/open?id=1Ahws2bBE_YSjvNcxsF9tnwRCv3HfaVhr))
  - ResNet-v2: Identity Mappings in Deep Residual Networks
  - Wide Residual Networks ([note](https://drive.google.com/open?id=14eQSeymwXgS7JvBbAkOnudAm6MFnJify), code)
* InceptionNet:
  - Inception-v1: GoogLeNet
  - Inception-v2, v3: Rethinking the Inception Architecture for Computer Vision ([note](https://drive.google.com/open?id=1SVOpf9aElrAGCZHlX7NvYL8pbehXpw8i), code)
  - Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning 
* DenseNet:
* NASNet: Learning Transferable Architectures for Scalable Image Recognition ([note](https://drive.google.com/open?id=1o1SfbVIgEhRWGQG_mPpxKoCDyWtNoifJ), code)
* EfficientNet:([note](https://drive.google.com/open?id=1LtdSId0HTpM8_O4k4WFrzHz4ldPf7dTu))


### [Visualizing CNNs](./doc/visualizing_cnn.md)
* DeconvNet
* BP: Deep inside convolutional networks: Visualising image classification models and saliency maps ([note](https://drive.google.com/open?id=1IBP1uMr08hBp3bKjvyNnwFMu0S8ORGcs))
* Guided-BP (DeconvNet+BP): Striving for simplicity: The all convolutional net ([note](https://drive.google.com/open?id=1KUq5-h_xVmjd4FudGDeBUfPV9vBMHV68), code)
* Understanding Neural Networks Through Deep Visualization


### [Weakly Supervised Localization](./doc/cam.md)
* From Image-level to Pixel-level Labeling with Convolutional Networks (2015)
* GMP-CAM: Is object localization for free? - Weakly-supervised learning with convolutional neural networks (2015) ([note](https://drive.google.com/open?id=1Xpnhq0snjkPMsxKLpmhLOpoZgfFlL9H3), code)
* GAP-CAM: Learning Deep Features for Discriminative Localization (2016) ([note](https://drive.google.com/open?id=1lrkE07E3bnLscAnScwq0OIO3AaRHrqnb), code)
* c-MWP: Top-down Neural Attention by Excitation Backprop
* Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2017) ([note](https://drive.google.com/open?id=10obbO7F2igia6gCcc9IqxATzOxdoPl7L), code)


### [Object Detection](./doc/detection.md)
* OverFeat - Integrated Recognition, Localization and Detection using Convolutional Networks ([note](https://drive.google.com/open?id=1O3j-ag0pPRbRjG4ovWmxnZIwhUJ0twvK), code)


### [Semantic Segmentation](./doc/semantic_segmentation.md)
* FCN_V1 (2014)에서 직접적인 영향을 받은 모델들:
  * FCN + max-pooling indices를 사용: SegNet V2 (2015) ([note](https://drive.google.com/open?id=1CDNkW-3LKVDjGAyPCgj8fOz78pMY0Pd7))
  * FCN 개선: Fully Convolutional Networks for Semantic Segmentation (FCN_V2, 2016) ([note](https://drive.google.com/open?id=1Kr2-ZdiqKmsgXP2ofaUZm_PT5UbbTyDN), [code](https://github.com/bt22dr/CarND-Semantic-Segmentation/blob/master/main.py))
  * FCN + atrous convolution과 CRF를 사용: DeepLap V2 (2016)
  * FCN + Dilated convolutions 사용: Multi-Scale Context Aggregation by Dilated Convolutions (2015)
  * FCN + Multi-scale: Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture (2015)
  * FCN + global context 반영 위해 global pooling 사용: ParseNet (2015) 
  * FCN + 모든 레이어에서 skip 사용: U-Net (2015) ([note](https://drive.google.com/open?id=1Up8PiwA79J8R3ScjgYTmLzt8pYYa83EN))
* PSPNet (2016) ([note](https://drive.google.com/open?id=1xPu7Z-0jWepxb1av9fG2Py72Yz0enWym))
* DeepLabv3+ (2018) ([note](https://drive.google.com/open?id=1YFUdcwKzIrTzfmL6o94y01tDXsZ2n6vc))
* EncNet (2018)
* FastFCN (2019)
* Instance Segmentation
  * DeepMask
  * SharpMask
  * Mask R-CNN (2017) ([note](https://drive.google.com/open?id=1kFVOdctJTcWYkflfCM1Ys-J7Fo8COC6R))
* 3D / Point Cloud
  * PointNet (2017)
  * SGPN (2017)
* Weakly-supervised Segmentation

### [Style Transfer](./doc/style_transfer.md)
* ~~A Neural Algorithm of Artistic Style (2015)~~
* Image Style Transfer Using Convolutional Neural Networks (2016)
* Perceptual Losses for Real-Time Style Transfer and Super-Resolution (2016)
* Instance Normalization: 
  * Instance Normalization: The Missing Ingredient for Fast Stylization (2016)
  * Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis (2017)


### Siamese, Triplet Network
* Triplet Network
  * FaceNet: A Unified Embedding for Face Recognition and Clustering ([note](https://drive.google.com/open?id=1E9ZGncIvpJoPK5_mSq5J-Mn33r2_xAqj), code)
  * Learning Fine-grained Image Similarity with Deep Ranking ([note](https://drive.google.com/open?id=1BrjRlzB139v5nmCgdLruJGn1drmBb33m), code)
* Siamese Network


### Mobile
* Shufflenet: An extremely efficient convolutional neural network for mobile devices
* Mobilenets: Efficient convolutional neural networks for mobile vision applications


### [Etc.](./doc/etc/md)
* A guide to convolution arithmetic for deep learning ([note](https://drive.google.com/open?id=1zGGzI4qc49u5zV0jFSkzD8xDMY0OalN1))


## [Generative Models](./doc/gan.md)

### Models

#### Autoregressive Models
* NADE (2011)
* RNADE (2013)
* MADE (2015)
* PixelRNN 계열
  * PixelCNN (2016): ([note](https://drive.google.com/open?id=1G_iIjf9dIWqge21sxrpcqK2L76PY8elN), [code1(mnist)](./code/PixelCNN_mnist.ipynb), [code2(fashion_mnist)](./code/PixelCNN_fashionmnist.ipynb))
  * WaveNet (2016) ([note](https://drive.google.com/open?id=1qnNQS_aFuPly8MVO7kSPytPAgf-KifbC), [code](./code/WaveNet.ipynb))
  * VQ-VAE: Neural Discrete Representation Learning

#### Variational Autoencoders
* VAE ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/doc/gan.md#auto-encoding-variational-bayes), [code1(mnist)](./code/vae.ipynb), [code2(fashion_mnist)](./code/vae_fashion_mnist.ipynb))
* Conditional VAE ([note](https://drive.google.com/open?id=1f9fGvvtj-FdPJRwtw7PFLW0ysAu-U_2O), [code](./code/conditional_vae_fashion_mnist.ipynb))
* VAE-GAN: Autoencoding beyond pixels using a learned similarity metric

#### Normalizing Flow Models
* NICE (2014) ([note](https://drive.google.com/open?id=1Bz8i8lASNr8SS6vBraDOyOPTJY61q2aG))
* Variational Inference with Normalizing Flows (2015) ([code](https://drive.google.com/drive/folders/1-kqyXOvnuw7aeOwbAfKO2OWUkFdv3AmR))
* IAF (2016)
* MAF (2017)
* Glow (2018)

#### GANs
* GAN: Generative Adversarial Networks ([note](https://drive.google.com/open?id=1gymav6NryH-0AJqJ7hRt6SzFLrX8bCIn), [code1(mnist)](./code/gan.ipynb), [code2(fashion_mnist)](./code/gan_fashion_mnist.ipynb))
* DCGAN ([note](https://drive.google.com/open?id=1IWeM32QDq97mQ8BdA-rWa58AiRqWTepG), [code1](./code/dcgan_mnist.ipynb), [code2](./code/dcgan_celebA.ipynb))
* WGAN 계열: 
  * WGAN: Wasserstein GAN ([note(진행중)](https://drive.google.com/open?id=1CnfvynSKj9apRZBLjzWB--QJ69PKj2wy), [code](./code/wgan.ipynb))
  * WGAN_GP: Improved Training of Wasserstein GANs 
  * CT-GAN: Improving the Improved Training of Wasserstein GANs
* infoGAN
* Improved GAN: 
* SNGAN: Spectral Normalization for Generative Adversarial Networks ([note(진행중)](https://drive.google.com/open?id=1qJmWsSKPQ2yXQDh68KcZsdEHeAkRvcgZ), [code](./code/sngan_fashion_mnist.ipynb))
* SAGAN: 
* CoGAN: Coupled Generative Adversarial Networks (note, code)

### [Image generation](./doc/img2img_translation.md)
#### image-to-image
* cGAN: Conditional Generative Adversarial Nets (2014) ([note](https://drive.google.com/open?id=1z1flvsqORItbCTZEGuFBQmubvNnAGzs-), [code](./code/cgan.ipynb))
* (내맘대로)pix2pix 계열:
  * pix2pix: Image-to-Image Translation with Conditional Adversarial Networks (2016) ([note](https://drive.google.com/open?id=1GYphdvvfuyb-YKDd_ItvwCLkNDqtY_5F))
  * ~~pix2pixHD~~
  * CycleGAN: 
  * BicycleGAN
  * vid2vid: Video-to-Video Synthesis
  * SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization (2019) ([note](https://drive.google.com/open?id=1CtbIRJ7wi7-wH9h7zQiClcHeqcfdoHEU))
* StarGAN: 
* PGGAN: 
* ~~UNIT/MUNIT~~
* ~~iGAN~~
* StyleGAN: 

#### text-to-image
* Generative adversarial text to image synthesis
* StackGAN
* AttnGAN

### Sequence generation
* WaveGAN: ([note](https://drive.google.com/open?id=1zd4pw884TztzisixmJSJDYTXcdXZSbdo), code)
* SeqGAN:

### Evaluation
* A note on the evaluation of generative models
* A Note on the Inception Score

### CS236 (Deep Generative Models)
* Introduction and Background ([slide 1](https://drive.google.com/open?id=1y9-nkh9OhxAuRP009FsZUDB8hjb3b9iG), [slide 2](https://drive.google.com/open?id=1Kmd7lnZJTw-mgwcTR91nWRVdxQ9Ot5X7))
* Autoregressive Models ([slide 3](https://drive.google.com/open?id=18l4h4iQ_lAROCOlKf44VGk-Q7DEvtBp6), [slide 4](https://drive.google.com/open?id=1IQ5LdSyO9UXi3yjh9c_m7AhK0jmIqQNF))
* Variational Autoencoders 
* Normalizing Flow Models 
* Generative Adversarial Networks 
* Energy-based models 

## NLP
* Recent Trends in Deep Learning Based Natural Language Processing ([note](https://drive.google.com/open?id=12dosro89x1wy3wXUfJ26oa3_2hU_S6n6))

### RNN Architecture
* Seq2Seq
  - Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation (2014)
  - Sequence to Sequence Learning with Neural Networks (2014) ([note](https://drive.google.com/open?id=1crgnXU-3JClMsF1ZYLr8bxxWkSZjcTEZ), code)
  - A Neural Conversational Model
* Attention
  - (Luong) Effective Approaches to Attention-based Neural Machine Translation (2015)
  - (Bahdanau) Neural Machine Translation by Jointly Learning to Align and Translate (2014) ([note](https://drive.google.com/open?id=1YJLljd9YbOzW5mADOtAT7V3imfMBi3Fg), code)
  - Transformer: Attention Is All You Need (2017)
* Memory Network
  - Memory Networks (2014) 
  - End-To-End Memory Networks (2015)
* Residual Connection
  - Deep Recurrent Models with Fast-Forward Connections for NeuralMachine Translation (2016)
  - Google's Neural MachineTranslation: Bridging the Gap between Human and Machine Translation (2016)
* CNN
  - Convolutional Neural Networks for Sentence Classification (2014)
  - ByteNet: Neural Machine Translation in Linear Time (2017)
  - Depthwise Separable Convolutions for Neural Machine Translation (2017)
  - SliceNet: Convolutional Sequence to Sequence Learning (2017)
### Word Embedding

## Multimodal Learning
* DeVise - A Deep Visual-Semantic Embedding Model: ([note](https://drive.google.com/open?id=19gr2FsgvfUAHHA4E25UFJAFmFSUD_L4k))
* Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models: ([note](https://drive.google.com/open?id=1qytxJgLsZvlbazeXTfExqPA-2rUX1irB))
* Show and Tell: ([note](https://drive.google.com/open?id=1fZJ7jopShsepyderJ03ivzoiGqMUOGl3))
* Show, Attend and Tell: ([note](https://drive.google.com/open?id=1COSvkFUWxuotzicGFAlLbl5l_iRrsTtG))
* Multimodal Machine Learning: A Survey and Taxonomy: ([note](https://drive.google.com/open?id=1qibjIoD5z6HjC_G6ICixpA_G0-A5P8t5))


## [Etc.](./doc/etc.md) (Optimization, Normalization, Applications)
* An overview of gradient descent optimization algorithms ([note](https://drive.google.com/open?id=1eSNr4zQBKbQQRpxDl06AWEwAU5qysTo_))
* Dropout:
* Batch Normalization: ([pdf+memo](https://drive.google.com/open?id=1rSM2Q510EjEZ3J6YpWH_ZwPiUS2JHQTp), code)
* How Does Batch Normalization Help Optimization?
* Spectral Norm Regularization for Improving the Generalizability of Deep Learning ([note(진행중)](https://drive.google.com/open?id=1_Th_cpo5rgTyQqi3085To_YgyNCmwlzL), code)
* Wide & Deep Learning for Recommender Systems
* Xavier Initialization - Understanding the difficulty of training deep feedforward neural networks
* PReLU, He Initialization - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification



## [Drug Discovery](./doc/medical.md)

