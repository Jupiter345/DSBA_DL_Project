# Hurricane Harvey Challenge: Semantic Image Segmentation of Aerial Imagery

## Authors
Anmol Katiyar (anmol.katiyar@student-cs.fr) <p>
Sarvani Sambaraju (sarvani.sambaraju@student-cs.fr)

## ABSTRACT
The Hurricane Harvey Challenge is a unique opportunity for researchers and practitioners in the field
of computer vision and natural disasters to advance the damage assessment process for post-disaster
scenarios using high-resolution UAS imagery. The dataset includes detailed semantic annotation of
damages, providing a valuable resource for training and evaluating models for image classification,
semantic segmentation. Several models of Semantic segmentation were used to determine the best fitted
model for the images. Out of all, PSPnet has given the highest accuracy.

## 1. PROBLEM DEFINITION AND DATA-PREPROCESSING
### Class imbalance: 
The dataset may be imbalanced, with more samples of one class than the other.
Techniques such as class weighting or oversampling to address this issue.
Clearly the classes are not balanced, it is evident that the property roof
class has been identified the most whereas there are not enough samples
for the class bridge.

### Synthetic Images: 
More data in the form of synthetic images can be
acquired to improve the performance of the model.

### Data augmentation: 
Data augmentation techniques such as random
rotations, horizontal and vertical flips, and cropping are used to
artificially increase the size of the training dataset and make the model
more robust to variations in the data. We have implemented several
parameters in the augmentation pipeline relating to the image details. The
OneOf block normalizes the probabilities of all augmentations inside it.
Imaging Warping technique like Grid distortion maps equivalent
families of curves which is combined with Optical Distortion ensures
that sections of the image gets distorted to the nearest pixel thereby using
Gaussian in procedure. The images are properly scaled and normalized before being fed into the model.

### Pre-Processing Images: 
Two functions have been implemented to one-hot encode the images. a class
vector (integers) is converted to a binary class matrix and then converted to a tensor.
Fine-tuning: Fine-tuning the pre-trained encoder weights by training the model for a few more epochs
with a lower learning rate. Learning rate was initially set to 0.0001. At epoch 38, it dropped to
2.417315480804104e-05.

### Hyperparameter tuning: 
Batch size was set to 4 after testing with multiples of 2. The Adaptive
gradient descent optimizer Adam ensures we do not have to change the learning rate.
Transfer learning: Since the dataset is relatively small, using a pre-trained model and fine-tuning it
on this dataset. This will allow to leverage the knowledge learned on a larger dataset. Prior datasets
like Imagenet pose a base model to transfer learn and improve the model. It also reduces the
processing time.

## 2. MODEL ARCHITECTURE
PSPNet (Pyramid Scene Parsing Network) is an extension of the DeepLabV3 architecture, which
uses a pyramid pooling module to extract features at multiple scales. The pyramid pooling module is a
convolutional layer that uses different pooling sizes to extract features at different scales. The advantage
of PSPNet is that it can handle large input images by using a smaller number of parameters, as a result
of the pooling layer, it also helps to improve the final results. Deeplabv3+ is an improvement over
DeepLabv3, which extends the model to include a "decoder" module. This module is used to upsample
the feature maps from the encoder and refine the segmentation results. This allows the model to produce
more detailed and accurate segmentation masks. Deeplabv3+ is an extension of the DeepLabv3 model,
which includes a decoder module, with the purpose of improving the final results.

<img width="821" alt="image" src="https://user-images.githubusercontent.com/29901358/218119884-3696e212-6abb-4b16-a179-00d306cb5dd7.png">

All the models were run with the same learning rate of 0.0001 and compared, the PSPNET with the
encoder vgg19 gave the best results. From the results in the table, it appears that the DeeplabV3+ model
performed the best out of the three models. It had the highest accuracy and Dice coefficient during
training and validation, as well as the lowest loss during training. This indicates that the DeeplabV3+
model was able to classify images correctly and segment them effectively.

The PSPNet model also performed well, with a relatively high accuracy and Dice coefficient during
training and validation, but it had a slightly higher loss during training compared to DeeplabV3+.
On the other hand, the Unet model had the lowest performance out of the three models. It had a relatively
low accuracy and Dice coefficient during training and validation, as well as a high loss during training.
This indicates that the Unet model struggled to classify images correctly and segment them effectively.
Overall, the results suggest that DeeplabV3+ is the best model for this specific task, followed by PSPNet
and then Unet.

## 3. EVALUATION
Main hyperparameters: The first callback is an exponential decay function for the learning rate, which
starts at a specified initial value (lr0) and decreases by a factor of 0.1 every s number of epochs. This is
intended to help the model converge faster and prevent overfitting. The second callback is a
ModelCheckpoint, which saves the model's weights to a specified file path at regular intervals or when
the performance metric (val_dice_coef) improves. The third callback is an EarlyStopping, which stops
training when the performance metric stops improving by at least 0.001 over a specified number of
epochs (patience=10). The fourth callback is a CSVLogger, which saves the training history to a csv
file. The fifth callback is a LearningRateScheduler, which applies the exponential decay function to the
learning rate during training.

An epoch refers to one complete iteration through the entire dataset during training. The results obtained
from the code are likely related to the performance of the model during training, specifically the loss,
accuracy, and dice coefficient for each epoch. Loss is a measure of how well the model is able to fit the
training data. It is a function that compares the predicted output of the model to the actual output and
calculates the difference. The goal of training is to minimize the loss. Accuracy is a measure of how
well the model is able to correctly predict the output for the training data. It is the percentage of correctly
classified examples in the training dataset. Dice coefficient is a measure of the similarity between the
predicted and actual output. It ranges between 0 and 1, with 1 indicating a perfect match and 0 indicating
no overlap. val_loss, val_accuracy, and val_dice_coefficient represent the same metrics, but calculated
on the validation dataset. These are used to evaluate the model's performance on unseen data, and are
used to check if the model is overfitting or underfitting.

Dice Loss calculates the sum of element-wise multiplication of the ground truth mask and the predicted
mask. The numerator in the equation represents the sum of all true positive pixels, which are the pixels
that are correctly classified as the object of interest. The denominator represents the sum of all pixels
that are either true positives or false positives. The final result is then negated and scaled by -1 to make
the function a loss function instead of a similarity metric. The results moving from -0.2755 to -0.8857
would mean that the model's performance got worse. A higher value of Dice loss indicates that the
predicted mask is less similar to the ground truth mask, meaning that the model's performance is worse.
Lower values indicate better performance, as the predicted and ground truth masks are more similar.

<img width="988" alt="image" src="https://user-images.githubusercontent.com/29901358/218119707-16b6aa13-f0db-498f-ae61-22fe56be176b.png">

## 4. CONCLUSION
In this architecture, the VGG19 encoder is used as a "feature extractor" to extract important features
from the input image. These features are then passed to the PSPNet, which uses a pyramid pooling
module and a convolutional neural network to perform the actual segmentation.This architecture is
commonly used because it allows the model to take advantage of the pre-trained VGG19 encoder,
which has already learned useful features from a large dataset of images. This can help the model to
perform better on the segmentation task, as it starts with a good set of features to work with.
Additionally, the PSPNet allows the model to perform the segmentation at different scales, which can
help to improve the overall performance.
Overall, the VGG19 encoder with PSPNet architecture is a powerful combination for image
segmentation tasks, as it combines the strengths of a pre-trained feature extractor with a specialized
segmentation network.

## 5. REFERENCES
[1] James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. 2013. An Introduction to Statistical Learning: With Applications
in R. Springer <p>
[2] Yurtkulu, S. C., Şahin, Y. H., & Unal, G. (2019, April). Semantic segmentation with extended DeepLabv3 architecture. In 2019 27th
Signal Processing and Communications Applications Conference (SIU) (pp. 1-4). IEEE <p>
[3] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. In Proceedings of the IEEE conference on computer
vision and pattern recognition (pp. 2881-2890).
