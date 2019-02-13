# deep-learning-papers

## Computer Vision
### CNN Architecture
* AlexNet: ImageNet Classification with Deep Convolutional Neural Networks
* ZFNet (DeconvNet): Visualizing and Understanding Convolutional Networks ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Visualizing%20and%20Understanding%20Convolutional%20Networks.pdf), code)
* NIN: Network in Network
* VggNet: Very Deep Convolutional Networks for Large-Scale Image Recognition
* GoogLeNet: Going Deeper with Convolutions
* ResNet:
  - ResNet-v1: Deep Residual Learning for Image Recognition
  - ResNet-v2: Identity Mappings in Deep Residual Networks
  - Wide Residual Networks ([note](./paper/Wide%20Residual%20Networks.pdf), code)
* InceptionNet:
  - Inception-v1: GoogLeNet
  - Inception-v2, v3: Rethinking the Inception Architecture for Computer Vision ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Rethinking%20the%20Inception%20Architecture%20for%20Computer%20Vision.pdf), code)
  - Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning 
* DenseNet:
* NASNet: Learning Transferable Architectures for Scalable Image Recognition ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/NASNET%20-%20Learning%20Transferable%20Architectures%20for%20Scalable%20Image%20Recognition.pdf), code)
* CapsNet:


### [Visualizing CNNs](./doc/visualizing_cnn.md)
* DeconvNet
* BP: Deep inside convolutional networks: Visualising image classification models and saliency maps ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Deep%20Inside%20Convolutional%20Networks-%20Visualising%20Image%20Classification%20Models%20and%20Saliency%20Maps.pdf), code)
* Guided-BP (DeconvNet+BP): Striving for simplicity: The all convolutional net ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Striving%20for%20Simplicity%20-%20The%20All%20Convolutional%20Net.pdf), code)
* Understanding Neural Networks Through Deep Visualization (paper)


### [Weakly Supervised Localization](./doc/cam.md)
* From Image-level to Pixel-level Labeling with Convolutional Networks (2015)
* GMP-CAM: Is object localization for free? - Weakly-supervised learning with convolutional neural networks (2015) ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Weakly-supervised%20learning%20with%20convolutional%20neural%20networks.pdf), code)
* GAP-CAM: Learning Deep Features for Discriminative Localization (2016) ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/CAM_%20Learning%20Deep%20Features%20for%20Discriminative%20Localization%20.pdf), code)
* c-MWP: Top-down Neural Attention by Excitation Backprop
* Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2017) ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Grad-CAM_%20Visual%20Explanations%20from%20Deep%20Networks%20via%20Gradient-based%20Localization.pdf), code)


### [Object Detection](./doc/detection.md)
* OverFeat- Integrated Recognition, Localization and Detection using Convolutional Networks ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/OverFeat-%20Integrated%20Recognition%2C%20Localization%20and%20Detection%20using%20Convolutional%20Networks.pdf), code)
* SPP
* R-CNN
  - R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation
  - Fast R-CNN
  - Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
  - Mask R-CNN
* SSD
* YOLO:
* YOLO9000:


### [Style Transfer](./doc/style_transfer.md)
* A Neural Algorithm of Artistic Style (2015)


### Semantic Segmentation
* FCN: Fully Convolutional Networks for Semantic Segmentation (note, code)

### Siamese, Triplet Network
* Triplet Network
  * FaceNet: A Unified Embedding for Face Recognition and Clustering ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/FaceNet%20-%20A%20Unified%20Embedding%20for%20Face%20Recognition%20and%20Clustering.pdf), code)
  * Learning Fine-grained Image Similarity with Deep Ranking ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Learning%20Fine-grained%20Image%20Similarity%20with%20Deep%20Ranking.pdf), code)
* Siamese Network


### Mobile
* Shufflenet: An extremely efficient convolutional neural network for mobile devices
* Mobilenets: Efficient convolutional neural networks for mobile vision applications


### [Etc.](./doc/etc/md)
* Image Captioning
* Show and Tell: Show and Tell: A Neural Image Caption Generator
* A guide to convolution arithmetic for deep learning ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/A%20guide%20to%20convolution%20arithmetic%20for%20deep%20learning.pdf))


## [Generative Models](./doc/gan.md)

### Models

#### Autoregressive Models
* PixelCNN: ([note](./paper/Pixel%20Recurrent%20Neural%20Networks.pdf), [code1(mnist)](./code/PixelCNN_mnist.ipynb), [code2(fashion_mnist)](./code/PixelCNN_fashionmnist.ipynb))
* PixelCNN++
* WaveNet

#### Latent Variable Models
* VAE ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/doc/gan.md#auto-encoding-variational-bayes), [code1(mnist)](./code/vae.ipynb), [code2(fashion_mnist)](./code/vae_fashion_mnist.ipynb))
* Conditional VAE ([note](./paper/Learning%20Structured%20Output%20Representation%20using%20Deep%20Conditional%20Generative%20Models.pdf), [code](./code/conditional_vae_fashion_mnist.ipynb))

#### GANs
* GAN: Generative Adversarial Networks ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Generative%20Adversarial%20Networks.pdf), [code1(mnist)](./code/gan.ipynb), [code2(fashion_mnist)](./code/gan_fashion_mnist.ipynb))
* DCGAN ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Unsupervised%20Representation%20Learning%20with%20Deep%20Convolutional%20Generative%20Adversarial%20Networks.pdf), [code1](./code/dcgan_mnist.ipynb), [code2](./code/dcgan_celebA.ipynb))
* WGAN: Wasserstein GAN ([note(진행중)](./paper/Wasserstein%20GAN.pdf), [code](./code/wgan.ipynb))
* Improved GAN: 
* ProgressiveGAN: 
* SNGAN: Spectral Normalization for Generative Adversarial Networks ([note(진행중)](./paper/Spectral%20Normalization%20for%20Generative%20Adversarial%20Networks.pdf), [code](./code/sngan_fashion_mnist.ipynb))
* SAGAN: 
* CoGAN: Coupled Generative Adversarial Networks (note, code)

### Image generation
* cGAN: Conditional Generative Adversarial Nets ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Conditional%20Generative%20Adversarial%20Nets.pdf), [code](./code/cgan.ipynb))
* pix2pix: 
* infoGAN: 
* CycleGAN: (note, code)
* BicycleGAN
* MUNIT
* iGAN

### NLP, Speech
* WaveGAN: ([note](./paper/WaveGAN-%20Synthesizing%20Audio%20with%20Generative%20Adversarial%20Networks.pdf), code)
* SeqGAN:

### Evaluation
* A note on the evaluation of generative models
* A Note on the Inception Score

### CS236 (Deep Generative Models)
* Introduction and Background ([slide 1](./paper/cs236_lecture1.pdf), [slide 2](./paper/cs236_lecture2.pdf))
* Autoregressive Models ([slide 3](./paper/cs236_lecture3.pdf), [slide 4](./paper/cs236_lecture4.pdf))
* Variational Autoencoders 
* Normalizing Flow Models 
* Generative Adversarial Networks 
* Energy-based models 

## NLP
* Recent Trends in Deep Learning Based Natural Language Processing ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Recent%20Trends%20in%20Deep%20Learning%20Based%20Natural%20Language%20Processing.pdf))

### RNN Architecture
* Seq2Seq
  - Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation (2014)
  - Sequence to Sequence Learning with Neural Networks (2014) ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks%20(2014).pdf), code)
  - A Neural Conversational Model
* Attention
  - (Luong) Effective Approaches to Attention-based Neural Machine Translation (2015)
  - (Bahdanau) Neural Machine Translation by Jointly Learning to Align and Translate (2014) ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate%20(2014).pdf), code)
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
* Show and Tell: ([note](https://onedrive.live.com/redir?resid=1450F7B8ED703390%21115&authkey=%21AJGFPawcTr22dxk&page=View&wd=target%28%EB%85%BC%EB%AC%B8.one%7C78936911-7ecb-1742-b29b-410f54136ea2%2FShow%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%7C2c22f00c-1325-6a40-955e-461a66e89d81%2F%29&wdorigin=703))
* Show, Attend and Tell: ([note](https://onedrive.live.com/redir?resid=1450F7B8ED703390%21115&authkey=%21AJGFPawcTr22dxk&page=View&wd=target%28%EB%85%BC%EB%AC%B8.one%7C78936911-7ecb-1742-b29b-410f54136ea2%2FShow%20Attend%20and%20Tell%20Neural%20Image%20Caption%20Generation%20with%20%7C71d6e739-bffb-f148-8674-55b32dc2681f%2F%29&wdorigin=703))



## Etc. (Optimization, Regularization, Applications)
* An overview of gradient descent optimization algorithms ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/An%20overview%20of%20gradient%20descent%20optimization%20algorithms.pdf))
* Dropout:
* Batch Normalization: ([note](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Batch%20Normalization-%20Accelerating%20Deep%20Network%20Training%20b%20y%20Reducing%20Internal%20Covariate%20Shift.pdf), code)
* Spectral Norm Regularization for Improving the Generalizability of Deep Learning ([note(진행중)](./paper/Spectral%20Norm%20Regularization%20for%20Improving%20the%20Generalizability%20of%20Deep%20Learning.pdf), code)
* Wide & Deep Learning for Recommender Systems



## Drug Discovery
* The rise of deep learning in drug discovery: https://doi.org/10.1016/j.drudis.2018.01.039

### De novo design
* RNN 계열
  * Generating Focussed Molecule Libraries for Drug Discovery with Recurrent Neural Networks
  * Generative Recurrent Networks for De Novo Drug Design
  * REINVENT: Molecular De Novo Design through Deep Reinforcement Learning
* GAN 계열
  * ORGAN
  * ORGANIC
* Graph 계열
  * Geometric deep learning: going beyond Euclidean data
  * Deeply learning molecular structure-property relationships using graph attention neural network

### Biological Imaging Analysis
* U-Net: Convolutional Networks for Biomedical Image Segmentation


