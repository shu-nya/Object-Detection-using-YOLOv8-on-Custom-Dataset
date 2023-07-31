# Object-Detection-using-YOLOv8-on-Custom-Dataset

## Objective

The goal is to detect guitars in images using YOLOv8 model. YOLOv8 is a state-of-the-art YOLO model that can be used for object detection, image classification, and instance segmentation tasks. YOLOv8 is fast, accurate, and easy to use. In this project, it has been used to detect guitars.

## Dataset

The dataset has been created by me. First, the copyright free images were collected from websites. Thereafter, they were annotated carefully using free labelling softwares available online. Attention was paid during labelling to maintain consistency of annotations. Cross-checking was done several times to avoid any discrepancy and ensuring the quality of the dataset.

The dataset has 2 folders:

1. train - contains 280 images. Out of these 280 images, 248 have one or more guitars in them, while the rest 32 images are only background having no guitar in them. These 32 background images are kept in the training dataset in order to penalize the False Positives. 'False Positives' (also known as 'Type I error') means that the model falsely makes a positive prediction when actually the condition (in this case, a guitar) does not exist. The background images (~ 10% of the size of the dataset) help the model in learning the case when a guitar is not present in the dataset. Hence, it penalizes the false positive and improves the Precision of the prediction.

2. val - contains 20 images. Care has been taken to prevent any data leakage. Any image present in the train set is not a part of the val set.

Hence, effort has been made to create a good quality dataset because the quality of the predictions depends a lot on the quality of the dataset the model has been trained on.

Structure of the dataset:

number of classes in the dataset = 1

classes = "Guitar"

Guitar_dataset

|__ train

----|______ images

----|______ labels

|__ val

----|______ images

----|______ labels

Note: Object detection requires large datasets. In this project, only in-built augmentation available with the YOLOv8 model during training has been used on the dataset. An analysis of the quality of augmentation has not been performed, as it is an iterative process and requires a lot of trial and error to get the right quality of augmentation. Also, the quality of augmentation plays a major role in the quality of predictions in object detection.

## Training

YOLOv8 has been custom trained to detect guitars. The training has been done in Google Colab by reading the dataset from Google Drive.

1. Set up the Google Colab
2. YOLOv8 Installation
3. Mount the Google Drive
4. Visualize the train images with their bounding boxes
5. Create the Guitar_v8.yaml (dataset config file) (YOLOv8 format)
6. Train the custom Guitar Detection model
7. Run Inference with the custom YOLOv8 Object Detector Trained Weights
8. Visualize the Predictions by plotting the validation images and their predicted bounding boxes

## Inference

On the validation set, the following values of metrics have been obtained:

1. Precision of Box = around 0.982
2. Recall of Box = around 1
3. mAP50 = around 0.995
4. mAP50-95 = around 0.874

Based on the evaluation metrics, the model is showing a decent performance which is also evident in the plotted images and their predicted bounding boxes. 

From the plot of the prediction, it is observed that the Recall is high (100%) as all the guitar images have been identified by the model. However, there is some issue of Precision, as in a couple of images, the model has made double predictions for the same guitar image. It may have been caused due to the side ways view of the guitar. Note that, care was taken to annotate the images of guitar only for the front view of guitar even in case the side view was visible along with the front view. This has been done to prevent any confusion for the model.

The model is able to get a good mAP value which is a widely used object detection metric for object detection models. This implies that the model is making a prediction of the bounding box that has a large Intersection over Union with the ground truth annotations provided with the validation images.

## Conclusion

Overall, the tasks performed in the project include creation of the dataset, then the object (guitar) detection, and finally the evaluation of the model's performance. The performance of the model can be improved by increasing the number and variety of the images in the dataset. Also, the number of classes can be increased by annotating the present images for other classes and also including more images and labelling them.

## Reference

https://github.com/ultralytics/ultralytics







