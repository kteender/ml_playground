from fastai.vision.all import *
matplotlib.rc('image', cmap='Greys')

from functools import reduce
from operator import concat
import os
import PIL

path = "/mnt/d/ml/ml_playground/KDEF/"
mini_path = "/mnt/d/ml/ml_playground/KDEF_mini/"
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
    suprised_ixs = [i for i in range(0, len(suprised_imgs))]
    sad_ixs = [i for i in range(0, len(sad_imgs))]

    valid_suprised_ixs = random.coices(suprised_ixs, k=int(len(suprised_ixs)/5))
    valid_sad_ixs = random.choices(sad_ixs, k=int(len(sad_ixs)/5))
    train_suprised_ixs = [i for i in range(0, len(suprised_ixs)) if i not in valid_suprised_ixs]
    train_sad_ixs = [i for i in range(0, len(sad_ixs)) if i not in valid_sad_ixs]

    valid_suprised_imgs = [suprised_imgs[i] for i in valid_suprised_ixs]
    valid_sad_imgs = [sad_imgs[i] for i in valid_sad_ixs]
    train_suprised_imgs = [suprised_imgs[i] for i in train_suprised_ixs]
    train_sad_imgs = [sad_imgs[i] for i in train_sad_ixs]

    train_suprised_tensors = [tensor(Image.open(o)) for o in train_suprised_imgs]
    train_sad_tensors = [tensor(Image.open(o)) for o in train_sad_imgs]
    valid_suprised_tensors = [tensor(Image.open(o)) for o in valid_suprised_imgs]
    valid_sad_tensors = [tensor(Image.open(o)) for o in train_sad_imgs]
    #These are rank 4 b/c they are color images
    train_stacked_suprised = torch.stack(train_suprised_tensors).float()/255
    train_stacked_sad = torch.stack(train_sad_tensors).float()/255
    valid_stacked_suprised = torch.stack(valid_suprised_tensors).float()/255
    train_stacked_sad = torch.stack(train_sad_tensors).float()/255
    """meansad = stacked_sad.mean(0)
    show_image(meansad)
    print(meansad.shape)
    plot = matplotlib.pyplot.imshow(meansad)
    #matplotlib.pyplot.show()
    #I'm not sure that the mean of all the images will work for my case?"""

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

    train_x = torch.cat([stacked_suprised, stacked_sad]).view(-1, 562*762*3)
    train_y = tensor([1]*len(sad_imgs) + [0]*len(suprised_imgs)).unsqueeze(1)
    valid_dset = list(zip(valid_x, valid_y))

    #the linear1 model in this chapter is a linear regression, so in the simple neural net
    #you do a linear regression in each training loop?
    #ml model != neural net
#simple_neural_net()

#chapter 5
#label the dataset
def get_label(s):
    return s[4:6]

def debug():
    for i in imgs:
        if "H." in i:
            print(i)

#softmax converts multiple activations to all be values between 0 and 1
#softmax is the first part of cross entropy loss, second part is log likelihood
def softmax(x): 
    return exp(x) / exp(x).sum(dim=1, keepdim=True)

def label():
    #for some reason batch_tfms is making the func stall out. Figure that out
    #I guess the DataBlock splits the dataset for you?
    faces = DataBlock(blocks = (ImageBlock, CategoryBlock),
                get_items=get_image_files, 
                splitter=RandomSplitter(seed=42),
                get_y=using_attr(get_label, 'name'),
                 #these two lines do presizing. You resize once, and than again to a
                 #smaller size to ensure that the transformed image in the batch
                 #won't have empty areas
                item_tfms=Resize(460),
                batch_tfms=aug_transforms(size=224, min_scale=0.75))
                
    dls = faces.dataloaders(path)
    dls.show_batch(nrows=1, ncols=3)
    #matplotlib.pyplot.show()
    #faces.summary(path)
    # I didn't really understand the section called Presizing, take a look at that
    #again
    x, y = dls.one_batch()
    print(y)
    #this will display the dependant variable, as integers. 
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(2)
    #this is a call to do the predicitons on just one label image set
    preds,_ = learn.get_preds(dl=[(x,y)])
    print(preds[0])
    #this is what we want, and we use the softmax to achieve it
    print(len(preds[0]),preds[0].sum())

def random_softmax_stuff():
    torch.random.manual_seed(42)
    acts = torch.randn((6,2))*2
    acts = torch.randn((6,2))*2

    #these two things will return the same
    #you're taking the softmax of the tensor
    (acts[:,0]-acts[:,1]).sigmoid()
    sm_acts = torch.softmax(acts, dim=1)

    

label()
#debug()
#I don't think this dataset it gonna work very well b/c of the profile faces
#remember log(a*b) = log(a)+log(b)