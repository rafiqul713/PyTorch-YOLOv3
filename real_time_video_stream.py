from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import time
import datetime
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import shutil

########################################################
###resolve conflict between opencv and ROS###
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
########################################################

import cv2





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    #print(opt)

    def convert_labels(x, y, w,h,shape):
        dw = 1.0/shape[0]
        dh = 1.0/shape[1]
        center_x=((x+y)/2.0)*dw
        center_y=((x+y)/2.0)*dh
        width = w*dw
        height = h*dh
        return (center_x,center_y,width,height)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    video_path= opt.video_folder
    video_fold=os.path.dirname(video_path)
    #print("Video path ",video_path)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frame_folder=video_fold+"/frame"
    if os.path.exists(frame_folder):
        shutil.rmtree(frame_folder)
    os.mkdir(frame_folder)
    Labels="output/labels"
    if os.path.exists(Labels):
        shutil.rmtree(Labels)
    os.mkdir(Labels)

    #image= cv2.resize(image,(416,416),cv2.INTER_LINEAR)
    counter = 0
    while success:
        if counter%5==0:
            cv2.imwrite(frame_folder+"/img_%d.jpg" % counter, image)     # save frame as JPEG file
            success,image = vidcap.read()
        counter= counter+1

    dataloader = DataLoader(
        ImageFolder(frame_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    #print("Data loader ",dataloader)
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        #print("Input image ",input_imgs)
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            #print("Detection ",detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    counter=0
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        label_path=Labels+"/img_"+str(counter)+".txt"
        file_open=open(label_path, "w")
        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        image_shape=img.shape
        #print("Image shape ",image_shape)
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                res=(convert_labels(float(x1),float(y1),float(box_w),float(box_h),image_shape))
                centerX,centerY,Width,Height = res
                #print("result ",centerX,centerY,Width,Height)
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                file_open.write(str(classes.index(classes[int(cls_pred)]))+" "+str(centerX)+" "+str(centerY)+" "+str(Width)+" "+str(Height)+"\n")

                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )


        counter=counter+5
        file_open.close()
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        #plt.savefig("detection/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.savefig("output/{}.jpg".format(filename), bbox_inches="tight", pad_inches=0.0)
        plt.close()
    shutil.rmtree(frame_folder)




