import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR(r'E:\Code\DD-DETR\main\runs\train\DD_DETR-300\weights\best.pt') # select your model.pt path
    model.predict(source=r'D:\Datasets\VisDrone2019\images\val',
                  conf=0.25,
                  project='runs/detect',
                  name='DD_DETR-300',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )