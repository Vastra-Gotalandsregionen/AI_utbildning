## Allmänna bibliotek
import pandas as pd
import torch
import torchvision
import numpy as np
import pdb
## Min egen datasetklass och funktioner
import dataset
## Bibiliotk för plottning
import matplotlib.pylab as plt
import plot_images
import plot_convergence


#################################################
### Data och dataloader
#################################################
print('1. Skapa datasets...')
# Läs in meetadata från fil till strukturen "dataframe"
metadata = pd.read_csv('/home/ida/projekt/Rontgenveckan_2022/data/data.csv')

# Dela upp datastrukturen i träning/validering
metadata_train = metadata[metadata.split=='train']
metadata_val   = metadata[metadata.split=='val']

# Augmenteringstransform
transform = torchvision.transforms.Compose([dataset.RandomScale()])

# Skapa datasetklasser att använda med dataloader
train_dset = dataset.my_dataset(metadata_train,transform=transform)
val_dset   = dataset.my_dataset(metadata_val)

# Skapa dataloaders
batch_size   = 10
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True,  num_workers=10)
val_loader   = torch.utils.data.DataLoader(val_dset,   batch_size=batch_size, shuffle=False, num_workers=10)

# Plotta en slumpmässig batch av bilder
#(img,label) = next(iter(train_loader))
#plot_images.show_grid(img,label,save=True)


#################################################
### Modell
#################################################
print('2. Skapa modell...')
# Förtränad modell från PyTorch
model = torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')

# Anpassa modell till 1 input-kanal och 3 output-klasser
conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
model._modules['conv1'] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model._modules['conv1'].weight.data = conv1
model.fc = torch.nn.Linear(512, 3)

# Number of model parameters
print('   Number of model parameters: {:.1e}'.format( sum(p.numel() for p in model.parameters() if p.requires_grad) ))

# Flytta modell till GPU
use_cuda = True
if use_cuda:
    model = model.cuda()


#################################################
### Förlustfunktion (loss function)
#################################################
print('3. Ansätt förlustfunktion...')
# Cross entropy förlust mellan label och input x: -x[label] + log( sum_j exp(x[j]) ). 
# Default: medelvärde över varje mini-batch vid loopning.
criterion = torch.nn.CrossEntropyLoss()

# Balanserad förlust, för olika storlek på klasserna {A, B, G}
# class_weights = torch.FloatTensor([1,3.3,10.6]).cuda()
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Flytta till GPU
if use_cuda:
    criterion = criterion.cuda()

#################################################
### Optimerare
#################################################
print('4. Ansätt optimerare...')
learning_rate = 0.001
optimizer     = torch.optim.SGD(model.parameters(), learning_rate,
                            momentum=0.9, nesterov=True, dampening=0, 
                            weight_decay=1e-4)


#################################################
### Snabba upp körningar med cuDNN
#################################################
# Auto-tuner som hittar bästa algoritmen för befintlig hårdvara
if use_cuda:
    torch.backends.cudnn.benchmark = True


#################################################
### Träna modell. Loopa över epoker
#################################################
print('5. Träna - loopa över epoker...')
# Ansätt output-fil med konvergensdata
convergence_file = 'results/convergence.csv'
fconv = open(convergence_file, 'w')
fconv.write('epoch,split,metric,value\n')
fconv.close()

# Träningsloop
epochs = 20
for epoch in range(1,epochs+1):
    
    # Sätt modellen till träningsläge
    model.train()
    
    # Initialisera förlust och noggrannhet för denna epok
    loss_train     = 0.
    accuracy_train = 0.
    
    # TRÄNINGSDATA: Loopa genom mini-batcher
    for i, (image,label) in enumerate(train_loader):
        # Flytta till GPU
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        
        # Break point
        #pdb.set_trace()
        
        # Framåtpass ger output sannolikheter
        output = model(image)
        
        # Normalisera så summan = 1 (sannolikheter)
        output = torch.softmax(output, dim=1)
        
        # Predikterad klass är den med max sannolikhet
        v, prediction = torch.max(output, dim=1)
        
        # Beräkna förlust för denna mini-batch
        loss_minibatch = criterion(output, label)
        
        # Sätt alla gradienter till noll
        optimizer.zero_grad()
        
        # Beräkna gradient för varje modellparameter: x.grad = x.grad + dloss/dx
        loss_minibatch.backward()
        
        # Stega optimerare för att uppdatera modellparametrar: x = x - lr*x.grad
        optimizer.step()
        
        # Spara förlust och nogrannhet för aktuell mini-batch storlek
        loss_train += loss_minibatch.item()*image.size(0)
        accuracy_train += torch.sum(prediction == label)
        
        # Skriv ut resultat för denna mini-batch
        print('BATCH [{}/{}]\t|  TRAIN  loss: {:.4f}\tacc: {:.4f}\t '.format(i+1, len(train_loader), loss_minibatch.item(),(prediction == label).float().mean().item()))
    
    # Medelvärde över hela datasetet
    loss_train = loss_train/len(train_dset)
    accuracy_train = accuracy_train/len(train_dset)
    
    
    # VALIDERINGSDATA: Loopa genom mini-batcher
    # Sätt modellen till utvärderingsläge
    model.eval()
    
    # Initialisera noggrannhet för denna epok
    accuracy_val = 0.
    
    for i, (image,label) in enumerate(val_loader):
        # Flytta till GPU
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        
        # Framåtpass ger output sannolikheter. 
        output = model(image)
        
        # Normalisera så summan = 1 (sannolikheter)
        output = torch.softmax(output, dim=1)
        
        # Predikterad klass är den med max sannolikhet
        _, prediction = torch.max(output, dim=1)
        
        # Spara nogrannhet för aktuell mini-batch storlek
        accuracy_val += torch.sum(prediction == label)
        
    # Medelvärde över hela datasetet
    accuracy_val = accuracy_val/len(val_dset)
    
    # Skriv ut resultat
    print('EPOK [{}/{}]\t|  TRAIN  loss: {:.4f}\tacc: {:.4f}\t |  VAL  acc: {:.4f}\n'.format(epoch, epochs, loss_train,accuracy_train,accuracy_val))
    
    # Skriv till fil
    fconv = open(convergence_file, 'a')
    fconv.write('{},train,loss,{}\n'.format(epoch, loss_train))
    fconv.write('{},train,accuracy,{}\n'.format(epoch, accuracy_train))
    fconv.write('{},val,accuracy,{}\n'.format(epoch, accuracy_val))
    fconv.close()
    
    # Spara aktuell modell till disk
    chpnt = {'epoch':      epoch,
             'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(chpnt, 'results/checkpoint.pth')  


#################################################
### Plotta konvergensdata
#################################################
plot_convergence.plot(save=True)
