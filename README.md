# 3D Object Generation with SP-GAN
![Blueno collage](/generator_outputs/collage/collage.png)
## Overview
This is my implementation of sphere-guided GAN (SP-GAN), a generative model recently proposed by Li et al. (2021), for my Brown Visual Computing starter project.
Trained on 3D point clouds of Blueno, our school's beloved but deceased big blue bear, the model produces unseen yet believable new instances of Blueno. 

At a high level, what differentiates SP-GAN from other GANs is its use of a global prior distribution in the form of a fixed sphere to regulate the holistic shape of the output *in addition to* a local prior (the standard Gaussian noise) that controls finer details. This allows the model to generate point clouds of complex shapes with little noise at the microscopic level. 

## Running the model
To use the trained model to produce Bluenos, fork the repo and run
```python inference.py```
Be sure to have the following dependencies installed:
- python 3(.9)
- tensorflow 2.7
- trimesh (install via pip trimesh[easy] for the required packages)
- matplotlib
- numpy

To train a model, run ```main.py```
## Model Architecture
The model contains several components worth highlighting. In the generator, the graph attention module is responsible for transforming the global sphere into a feature map, which is then normalized, per instance, via the local features computed from the latent vector. The attention module borrows heavily from DGCNN's EdgeConv operation (Wang et al. 2019) by grouping nearby points through the k-nearest neighbors algorithm and passing each group through MLPs. To produce the final output, the features are passed through several MLPs consisting of repeated conv2d, LeakyReLU, and batch normalization.

On the other hand, the discriminator is far simpler; after a series of convolutions *two* categories of confidence score are outputted, one for the shape holistically and the other for each point in the shape's cloud. These are combined in the MSE loss function.

Below are diagrams of the high-level architecture of the generator and discriminator from the paper, respectively, where S represents the point cloud for the fixed sphere, <img src="https://latex.codecogs.com/gif.latex?z " /> is a duplicated latent noise vector, <img src="https://latex.codecogs.com/gif.latex?P " /> is the generator output and <img src="https://latex.codecogs.com/gif.latex?\hat{P} " /> is a real data sample.
![generator_architecture](/generator_architecture.png)
![discriminator_architecture](/discriminator_architecture.png)

## Dataset
I coded a procedure in OpenSCAD to generate random, fully parametrized Bluenos. Dozens of features are randomly determined during compilation, ranging from Blueno's headlamp length/width to its leg radius/length/spread angle. I manually set the parameter bounds so that no grossly deformed or genus-1+ Bluenos were birthed.  Here are samlpes of the 3D Blueno models:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/blueno/blueno0.png" width="300"> <img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/blueno/blueno1.png" width="350"> <img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/blueno/blueno2.png" width="300">

After rendering, the Bluenos are exported as STL meshes and imported in main.py, where they are converted into point clouds via random point sampling done by trimesh. These point clouds are used as the input into the discriminator. Below is an example of a converted input cloud. Note the uniformity of the sampling.

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/blueno/input_cloud.gif" width="700">

## Training
The GAN was trained adversarially on ~1000 examples for 80 epochs with a learning rate of 0.0001, batch size of 16 and the following model hyperparameters
- latent vector dimension = 100
- number of points in a cloud = 2048
- per-point confidence weight = 0.4 (for loss function)
- per-shape confidence weight = 1
- optimizer = adam (momentum=0.5)

## Results
Below are output samples produced by the trained generator.

Typical Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/blueno1.gif" width="700">

Thick-armed, longer-nosed Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/blueno2.gif" width="700">

Long-legged, thick-armed blueno

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/blueno3.gif" width="700">

Asymmetrical, flat-headed, (very) long-lamped, wide-eared, thicker-armed, thin/long-legged Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/blueno4.gif" width="700">

Asymmetrical, thin, short-lamped, short-armed, short-legged Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/blueno5.gif" width="700">


### Interpolation
Now, as no GAN project is complete without a latent space interpolation, here's your daily dose (notice the proportions of the arms, legs, and lamp):

Source Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/interpolation/1.gif" width="700">
<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/interpolation/2.gif" width="700">
<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/interpolation/3.gif" width="700">

Target Blueno:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/final/interpolation/4.gif" width="700">

The generated blueno point clouds bear impressive resemblance to their real CAD siblings, ~~pun perhaps intended~~. The model is able to faithfully capture the macro shape of blueno as well as finer features with high granularity given the relatively limited training time. Surfaces traced by the points are smooth (observe the smooth pear-shaped body and well-formed head) and edges are sharp (e.g. lamp hemisphere). Upon closer inspection, however, it can be seen that noise is most apparent around parts with the most variability in the input data - notably the arms and legs.

It is evident that the model did not simply memorize training examples despite the very close shape resemblance to the training data; the points sampled by trimesh were largely uniform across the input mesh surfaces whereas the point density in the generated outputs varied noticeably. Furthermore, manipulating the latent noise vector resulted in bluenos with asymmestric features that were, by design, never present in the training data. This indicates that the model was capable of learning a high-level volumetric representation of blueno without resorting to point memorization. Though unlikely, it could be argued that this is a result of the limited training and that output points may converge to those on the seen input clouds given enough training epochs (80 vs. 2000 as conducted by the authors).

While exploring the latent space, I was able to find healthy diversity in the Blueno output population, though finding the dimension(s) corresponding to specific features of Blueno was difficult; the features seemed entangled across several different dimensions. Despite this, it is at least clear that mode collapse did not occur.

## The journey and its obstacles
### 1) Decrypting the architecture
As this was my first attempt beyond a vanilla GAN, I initially struggled to fully understand SP-GAN and its inner-workings. Decrypting the paper required me to read the related works from which the authors drew their ideas and modules (especially DGCNN and PointNet), and then the works which those works' authors drew from, etc.  But doing so gave me a much stronger understanding of the landscape of deep 3D generative models and their origins.

### 2) Co-opting modules
Trying not to reinvent the wheel, I co-opted a few helper modules from the related works such as the code for constructing the kNN graph used in the generator. However, tweaking the code for my needs required understanding uncommented hieroglyphics and took longer than I had expected. 

### 3) Making ground-truth Bluenos
Surprisingly, writing the actual procedure to generate random Bluenos was far less frustrating than actually rendering and exporting the Bluenos for use. Since OpenSCAD has no option to export models within the code, generating 1000 Bluenos would have required manual labor using the GUI. However, after discovering that OpenSCAD has command line export commands, I wrote a python script to generate a Windows batch file which repeatedly calls the cmd commands.

### 4) Training + hyperparameter tuning
After getting locked out of Google Colab by reaching usage limits only to realize that my initial hyperparameters caused a failure mode, I was forced to train my model on my baked potato (Intel i5 laptop, no dedicated graphics). I made the decision not to augment my training data via translations and rotations because doing so would require significant training time and RAM to acheive the desired  transformation invariances. Tuning the hyperparameters was also a balancing act between getting recognizable outputs and staying under hardware limitations. I frequently encountered system freezes and out-of-memory errors which were somewhat alleviated with smaller batches, but it is fortunate that no electronics were set ablaze in the day-long training attempts. 


## Training images for your viewing pleasure
Untrained generator:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/epoch0.png" width="500">

3 epochs:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/epoch3.png" width="500">

10 epochs:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/epoch10.png" width="500">

20 epochs:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/epoch20.png" width="500">

40 epochs:

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/epoch40.png" width="500">

Fully trained (75 epochs):

<img src="https://github.com/dinhanhtruong/3D-Object-Generation-with-SP-GAN/blob/main/generator_outputs/progress_pics/final.png" width="500">

## References
[1] Ruihui Li, Xianzhi Li, Ka-Hei Hui, and Chi-Wing Fu. 2021. SP-GAN: Sphere-Guided 3D Shape Generation and Manipulation. ACM Trans. Graph. 40, 4,
Article 151 (August 2021), 13 pages. https://doi.org/10.1145/3450626.3459766

[2] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M.
Solomon. 2019. Dynamic graph CNN for learning on point clouds. ACM Transactions
on Graphics 38, 5 (2019), 146:1–146:12.
