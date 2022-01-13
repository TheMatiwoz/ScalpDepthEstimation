# ScalpDepthEstimation

Repository created for the purpose of realization of an engineering thesis, the topic of which is "An implementation of an algorithm for depth estimation from monocular video in medical applications". The project includes an implementation of the training phase, a model and an application. The training data consists of artificially generated scalp recordings and corresponding SfM reconstruction results. 
The PyTorch framework was used to implement the model. A summary of the engineering work is presented below.

## Abstarct

The purpose of this thesis is to propose a method and implement an algorithm for depth estimation on scalp recordings. The described solution uses deep learning techniques implemented
in an unsupervised manner. The only data that need to be fed into the model based on convolutional neural networks are the scalp recordings and the corresponding spatial reconstruction results
obtained from the Structure-from-Motion method. This means no need to generate and label the
data manually to obtain reference depth maps. Both training and validation data were generated in
Blender software and simulate real recordings. The training architecture consists of two branches - it forms a Siamese network. The model consists of several layers. The first is a modified FCDenseNet57 architecture. The next uses data from the SfM reconstruction and allows for correct
parameter optimization. For model evaluation, matching depth maps were generated and used
as ground truth. This enabled the use of mean squared error and peak signal-to-noise ratio. The
obtained results on synthetic data show good potential of the method. In this thesis, the negative
effect of homogeneous object texture on the reconstruction results and thus on the overall model
performance is also addressed. An application has been developed that allows depth prediction
to be made from video sequences.
