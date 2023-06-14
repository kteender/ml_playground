#imports
from fastai.vision.all import *
matplotlib.rc('image', cmap='Greys')

from functools import reduce
from operator import concat

#matplotlib.rc('image', cmap='Greys')

def refuse_pile():
    #pytorch nn.Linear does the same thing as init_params and linear together -- combines 
    #both weights and biases into a single class

    #Creates a tensor of a given sized with random vals from a normal distribution
    #with a mean of 0 and a standard deviation of std
    def init_params(size, std=1.0):
        return (torch.randn(size)*std).requires_grad_()
    
    weights = init_params((28*28,1))
    bias = init_params(1)

class BasicOptim:
    def __init__(self,params,lr): 
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: 
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: 
            p.grad = None

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

#making a dict where key is digit, value is list of shit that corresponds to digit
#0 - images
#1 - a tensor with all the images in the directory
def make_dataset(path):
    num_dict = {}
    for i in range(0,10):
        imgs = (path/'training'/str(i)).ls().sorted()
        #get some validation images 
        valid = int(len(imgs)/10)*-1
        training_imgs = imgs[:valid]
        validation_imgs = imgs[valid:]
        training_img_tens = [tensor(Image.open(o)) for o in training_imgs]
        valid_img_tens = [tensor(Image.open(o)) for o in validation_imgs]
        #num_img_tens = [tensor(Image.open(o)) for o in imgs]
        training_stacked_tens = torch.stack(training_img_tens).float()/255
        valid_stacked_tens = torch.stack(valid_img_tens).float()/255
        #num_stacked_tens = torch.stack(num_img_tens).float()/255
        num_dict[i] = [training_imgs, training_img_tens, training_stacked_tens, validation_imgs, valid_img_tens, valid_stacked_tens]

    return num_dict

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
# weights.grad.shape,weights.grad.mean(),bias.grad

#training loop
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

#accuracy func
def batch_accuracy(xb, yb):
    preds = xb
    corrects = preds.float() == yb
    return corrects.float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    #this line makes a new tensor of the values in accs, makes an average of all 
    #the values, which will return a single-item tensor. The four rounds the num
    #to four decimal places
    return round(torch.stack(accs).mean().item())
                 
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

def main():

    #MNIST Loss Function
    #create a tensor of all the images
    #path = untar_data(URLs.MNIST)
    #Path.Base_PATH = path
    path = Path('/home/kteender/.fastai/data/mnist_png/')
    print(path.ls())

    num_dict = make_dataset(path)
    train_x = torch.cat([num_dict[k][2] for k in num_dict.keys()]).view(-1, 28*28)
    #create corresponding labels. The multiplication creates a new list that repeats the first number len(imgs) times
    #The little reduce incantation in there flattens the out the outer layer of a multi-dimensional list
    train_y = tensor(reduce(concat, [[k]*len(num_dict[k][0]) for k in num_dict.keys()])).unsqueeze(1)
    #This combines the two tensors into tuples of item and labels
    dset = list(zip(train_x, train_y))

    valid_x = torch.cat([num_dict[k][5] for k in num_dict.keys()]).view(-1, 28*28)
    valid_y = tensor(reduce(concat, [[k]*len(num_dict[k][3]) for k in num_dict.keys()])).unsqueeze(1)
    valid_dset = list(zip(valid_x, valid_y))

    dl = DataLoader(dset, batch_size=256)
    valid_dl = DataLoader(valid_dset, batch_size=256)
    xb, yb = first(dl)
    print(xb.shape)
    print(yb.shape)

    lr = .00001
    linear_model = nn.Linear(28*28,1)

    use_nonlinear = False
    if use_nonlinear:
        #initilize an optimizer. SGD in fastai == Basic Opt
        opt = SGD(linear_model.parameters(), lr)
        
        #train the model. == Learner.fit
        epochs = 20
        for i in range(epochs):
            #train an epoch
            for xb, yb in dl:
                calc_grad(xb, yb, linear_model)
                opt.step()
                opt.zero_grad()
            accs = [batch_accuracy(linear_model(xb), yb) for xb,yb in valid_dl]
            #this line makes a new tensor of the values in accs, makes an average of all 
            #the values, which will return a single-item tensor. The four rounds the num
            #to four decimal places
            accuracy = round(torch.stack(accs).mean().item())
            print(accuracy)
            print("\n")

    #This arch just has one nonlinearity. The first and third lines below are layers
    #The second line is a nonlinearity, aka activation function, that determines 
    #from the outputs of the neurons in the first layer whether the second layers neurons 
    #is activated

    #Each layer needs to have the same number of neurons -- 30 in this case
    #last layer needs to have 10 neurons for each digit
    simple_net = nn.Sequential(
        nn.Linear(28*28,30),
        nn.ReLU(),
        nn.Linear(30,10))
    
    dls = DataLoaders(dl, valid_dl)
    
    learn = Learner(dls, simple_net, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
    learn.fit(40, 0.1)

    lr = .001

    plt.plot(L(learn.recorder.values).itemgot(2))
    matplotlib.pyplot.show()
    learn.recorder.values[-1][2]
    #opt =

    """
    #This is an example line of the inputs for one image
    #calculate the weighted sum of inputs to the neuron. The weighted sum is the relative importance
    #of each element for the final result
    (train_x[0]*weights.T).sum() + bias
    #Here is a function that does the same thing
    def linear1(xb): 
        return xb@weights + bias
    preds = linear1(train_x)  
    corrects = preds.float() == train_y
    """

main()
