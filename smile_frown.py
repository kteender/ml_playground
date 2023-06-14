from fastai.vision.all import *
matplotlib.rc('image', cmap='Greys')

from functools import reduce
from operator import concat
import os
import PIL

path = "/mnt/d/ml/ml_playground/KDEF/"
save_path = "/mnt/d/ml/ml_playground/"
imgs = []
for f in os.listdir(path):
    for i in os.listdir(os.path.join(path, f)):
        imgs.append(os.path.join(path, f, i))
#imgs = reduce(concat, [os.listdir(os.path.join(path, f)) for f in os.listdir(path)])

#Lesson 1
def is_happy(img):
    exp_label = img.split("\\")[-1][4:6]
    #print(exp_label)
    return exp_label == 'HA'

def simple_model():
    #img_paths = reduce(concat, [os.path.join(path, f, i) for i in os.listdir(os.path.join(path, f)) for f in os.listdir(path)])
    dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_happy, item_tfms=Resize(224))

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    print(learn)
    uploader = SimpleNamespace(data = ['/mnt/d/ml/ml_playground/frown.JPG'])
    img = PILImage.create(uploader.data[0])
    results, _, probs = learn.predict(img)
    print(f"Is this happy?: {results}.")
    print(f"Probability it's happy: {probs[1].item():.6f}")

#print(is_happy('D:\ml\ml_playground\KDEF\AF01\AF01HAS.JPG'))

# never finished this function. Lesson 2
def data_clean():
    failed = verify_images(imgs)
    print(failed)
    faces =  DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y = parent_label,
        item_tfms=Resize(128)
    )
    dls = faces.dataloaders(path)
    #dls.va
    #with this dataset, everything passes

#Lesson 4
def suprised_sad_images(images):
    suprised = [img for img in images if img.split('/')[-1][4:6] == 'SU']
    sad = [img for img in images if img.split('/')[-1][4:6] == 'SA']
    return suprised, sad

def simple_neural_net():
    suprised_imgs, sad_imgs = suprised_sad_images(imgs)
    suprised_tensors = [tensor(Image.open(o)) for o in suprised_imgs]
    sad_tensors = [tensor(Image.open(o)) for o in sad_imgs]
    #These are rank 4 b/c they are color images
    stacked_suprised = torch.stack(suprised_tensors).float()/255
    stacked_sad = torch.stack(sad_tensors).float()/255
    meansad = stacked_sad.mean(0)
    show_image(meansad)
    print(meansad.shape)
    plot = matplotlib.pyplot.imshow(meansad)
    #matplotlib.pyplot.show()
    #I'm not sure that the mean of all the images will work for my case?
    #you've got issues b/c of how the color images have 3 channels
    #the train_x tensor 
    print("concatonated tensors shape")
    print(torch.cat([stacked_suprised, stacked_sad]).shape)
    #this tensor has all the channels of the image combined w/ the dimens
    train_x = torch.cat([stacked_suprised, stacked_sad]).view(-1, 562*762*3)
    train_y = tensor([1]*len(sad_imgs) + [0]*len(suprised_imgs)).unsqueeze(1)
    print("Final shapes")
    print(train_x.shape)
    print(train_y.shape)
    dset = list(zip(train_x, train_y))
    x, y = dset[800]
    print(x.shape, y)
simple_neural_net()


