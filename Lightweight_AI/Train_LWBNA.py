import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2,os, glob
from Lightweight_AI import LWBNA_models
from tensorflow import keras

# This function is to Normalize the image intensity between 0 and 255
def Normalize_0_255(image):
    image = image-np.amin(image)
    image = image/np.amax(image)
    image = 255*image
    image = image.astype(np.uint8)
    return(image)
# This function is to equalize the image intensity

def Image_intensity_equalize(image, clip_limit):
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Input is not a valid image")

        image = (image - np.mean(image)) / np.std(image)
        image = Normalize_0_255(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image_planes = np.array(cv2.split(image))
        CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=((8, 8)))
        image_planes[0] = CLAHE.apply(image_planes[0])
        image = cv2.merge(image_planes);image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        image = Normalize_0_255(image)
    except TypeError as e:
        print(f'Error: {e}')
        return None
    return image

# This function assumes both training image and GT/Mask image have the same filename and image type (.png or .jpg etc)
def Load_segmentation_training_dataset(img_size = (512,512), Resize=False,img_preprocess=False, title='Load training'):
    root = tk.Tk()
    file_image_dir = filedialog.askopenfilename(title=title +' image directory:')
    file_GT_image_dir = filedialog.askopenfilename(title=title + '/GT image directory:')
    root.destroy()
    image_dir,file = os.path.split(file_image_dir); GT_image_dir, _ = os.path.split(file_GT_image_dir)
    file_type=file[len(file)-4:]
    images_data = [];GT_images=[] # Lists to store the images and corresponding mask image
    counter=0
    for img_path in sorted(glob.glob(image_dir+ '/*' + file_type), key=os.path.basename):  # This for loop is for reading image files
        GT_flag=False
        f_name = img_path[len(image_dir)+1:len(img_path) - 4]  # File extention is not included
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert the format to RGB
        try:
            GT_img = cv2.imread(GT_image_dir + '/'+f_name + file_type, 1)
            GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2RGB)  # convert the format to RGB
            GT_flag =True
        except:
            print('GT image file is not found: ', f_name)
            pass
        if GT_flag == True:
            print('%d. %s'%(counter,f_name))
            if img_preprocess == True:
                img = Image_intensity_equalize(img, clip_limit=1.0)
            if Resize == True:
                img = cv2.resize(img, img_size)
                GT_img = cv2.resize(GT_img, img_size)
            cv2.imshow('Image', img); cv2.imshow('GT', GT_img);cv2.waitKey(1)
            images_data.append(img);GT_images.append(GT_img)
            counter +=1
    print('Total number of images loaded: ', counter)
    images_data = np.array(images_data); GT_images = np.array(GT_images)
    cv2.destroyAllWindows()
    img_train=images_data.astype('float32')/255; GT_train = GT_images.astype('float32') / 255
    img_view=images_data[0,:,:,:]; GT_img_view = GT_images[0, :, :, :]
    img_train = np.reshape(img_train, (len(img_train), img_view.shape[0], img_view.shape[1], img_view.shape[2]))
    GT_train = np.reshape(GT_train,(len(GT_train), GT_img_view.shape[0], GT_img_view.shape[1], GT_img_view.shape[2]))
    return (img_train, GT_train)  # It returns the image data with intensity 0-255


def Train_LWBNA_segmentation_model(image_size=(512,512),Resize=False,img_preprocess=False,total_epochs=500,batch_size=4,
                    model_name='LWBNA_unet',save_path='', load_validation_data = True, early_stop=True, patience=40):
    np.random.seed(42)
    X_train, y_train = Load_segmentation_training_dataset(img_size = image_size, Resize=Resize,
                                                img_preprocess=img_preprocess, title='Load any image from training')
    if load_validation_data:
        X_valid, y_valid = Load_segmentation_training_dataset(img_size = image_size, Resize=Resize,
                                                img_preprocess=img_preprocess, title='Load any image from validation')
    else:
        t_imgs = int(0.2*X_train.shape[0])
        print('Number images used for validation from the training dataset are: ', t_imgs)
        X_valid = X_train[:t_imgs, :, :, :]; y_valid = y_train[:t_imgs, :, :, :]

    # These are the 10 images from validation set to test trained model prediction
    testing_images = X_valid[1:11, :, :, :]; test_GT = y_valid[1:11, :, :, :]
    # This is to confirm the loaded image and it GT/mask
    img1 = testing_images[0, :, :, :]; GT1 = test_GT[0,:,:,:]
    img1 = Normalize_0_255(img1); GT1=Normalize_0_255(GT1)
    cv2.imshow("Test image",img1);cv2.imshow("Test GT",GT1);cv2.waitKey()
    cv2.destroyAllWindows()

    img_shape = img1.shape
    LWBNA_seg_model = LWBNA_models.LWBNA_unet_segmentation_model(img_shape)
    losses = {'Seg_out': LWBNA_models.dice_p_bce}
    metrics = {'Seg_out': LWBNA_models.dice_coef}
    opt = keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
    LWBNA_seg_model.compile(optimizer=opt, loss=losses, metrics=metrics)
    train_targets = {'Seg_out': y_train}
    validation_targets = {'Seg_out': y_valid}
    if early_stop:
        early_stoping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-8,
                                                                                    restore_best_weights=True)
        history = LWBNA_seg_model.fit(X_train, train_targets, validation_data=(X_valid, validation_targets),
                                  epochs=total_epochs,batch_size=batch_size, shuffle=True, callbacks=[early_stoping])
    else:
        history = LWBNA_seg_model.fit(X_train, train_targets, validation_data=(X_valid, validation_targets),
                                                            epochs=total_epochs, batch_size=batch_size, shuffle=True)
    #Save the trained model and training curves
    acc_data = pd.DataFrame(history.history)
    if save_path !='':
        # Create the directory to save the trained model data
        checkpoint_dir = save_path + '/' + model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        LWBNA_seg_model.save(checkpoint_dir + '/' + model_name + '.h5')
        acc_data.to_csv(checkpoint_dir + '/' + model_name + '.csv',index=True,header=True)  # write data to excel file
    else:
        LWBNA_seg_model.save(model_name + '.h5')
        acc_data.to_csv(model_name + '.csv', index=True, header=True)  # write data to excel file

    pd.DataFrame(history.history).plot(figsize=(8, 5));plt.grid(True);plt.show()

    # predict the segmentation mask by trained LWBNA_unet model
    pred = LWBNA_seg_model.predict(testing_images)
    for i in range(0, 9, 1):
        image = testing_images[i, :, :, :];image = Normalize_0_255(image)
        GT_img = test_GT[i, :, :, :]; GT_img = Normalize_0_255(GT_img)
        pred_mask = pred[i, :, :, :]; pred_mask = Normalize_0_255(pred_mask)
        cv2.imshow('Input image',image); cv2.imshow('GT_mask',GT_img); cv2.imshow('Predicted_mask',pred_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Train_LWBNA_segmentation_model(image_size=(320,320),Resize=False,img_preprocess=False,total_epochs=500,batch_size=4,
            model_name='LWBNA_unet',save_path='', load_validation_data = False, early_stop=True, patience=40)
