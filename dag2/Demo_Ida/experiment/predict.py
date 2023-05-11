import pandas as pd
import torch
import torchvision
import numpy as np
import dataset
import matplotlib.pylab as plt
import plot_images

#################################################
### Data och dataloader
#################################################
print('1. Skapa dataset...')
# Läs in meetadata från fil till strukturen "dataframe"
metadata = pd.read_csv('/home/ida/projekt/Rontgenveckan_2022/data/data.csv')

# Dela upp datastrukturen i test
metadata_test  = metadata[metadata.split=='test']

# Skapa datasetklass att använda med dataloader
test_dset  = dataset.my_dataset(metadata_test)

# Skapa dataloader
batch_size   = 10
test_loader  = torch.utils.data.DataLoader(test_dset,  batch_size=batch_size, shuffle=False, num_workers=10)

# (img,label) = next(iter(test_loader))
# plot_images.show_grid(img,label)

#################################################
### Modell
#################################################
print('2. Ladda modell...')
# Modell från PyTorch
model = torchvision.models.resnet34()

# Anpassa modell till 1 input-kanal och 3 output-klasser
conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
model._modules['conv1'] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model._modules['conv1'].weight.data = conv1
model.fc = torch.nn.Linear(512, 3)
model = model.cuda()

# Ladda tränade vikter
chpnt = torch.load('results/checkpoint.pth')
model.load_state_dict(chpnt['state_dict'])

#################################################
### Prediktera
#################################################
print('3. Prediktera klasser...')

# Sätt modellen till utvärderingsläge
model.eval()

# Initialisera
accuracy_test = 0.
label_pred    = np.empty(len(test_dset))

# TESTDATA: Loopa genom mini-batcher
for i, (image,label) in enumerate(test_loader):
    # Copy to GPU
    image = image.cuda()
    label = label.cuda()
    
    # Framåtpass ger output sannolikheter
    output = model(image)
    _, prediction = torch.max( torch.softmax(output, dim=1), dim=1 )
    
    # Spara noggrannhet
    accuracy_test += torch.sum(prediction == label)
    label_pred[i*batch_size:i*batch_size+image.size(0)]= prediction.cpu().numpy()

# Medelvärde över hela datasetet
accuracy_test = accuracy_test/len(test_dset)

# Plotta slumpade exempel
test_loader2  = torch.utils.data.DataLoader(test_dset,  batch_size=21, shuffle=True, num_workers=10)
(img,label) = next(iter(test_loader2))
# label_new = [str(x)+' / {:.0f}'.format(y) for x,y in zip(label.numpy(),label_pred[0:len(label)])]
label_new = ['True:'+str(x)+'\nPred:{:.0f}'.format(y) for x,y in zip(label.numpy(),label_pred[0:len(label)])]
plot_images.show_grid(img,label_new,figno=2)

# Analysera resultat
df = test_dset.dataframe[['Subject_ID', 'Sex', 'subtype', 'label', 'split']].copy()
df['pred'] = label_pred.astype(int)
df['correct'] = df.pred == df.label
print('\n',df[['Subject_ID','subtype','label','pred','correct']])

results = df.groupby('subtype').correct.value_counts(normalize=0,sort=False).unstack(fill_value=0).stack().reset_index().rename(columns={0:'counts'})
results2 = df.groupby('subtype').correct.value_counts(normalize=1,sort=False).unstack(fill_value=0).mul(100).round(0).astype(int).stack().reset_index().rename(columns={0:'percentage'})
results = results.merge(results2,on=['subtype','correct'])
print('\n',results)
print('\n',df.groupby('subtype').correct.mean().reset_index())

print('\nTEST  accuracy:          {:.4f}'.format(df.correct.mean().mean()))
print('TEST  balanced accuracy: {:.4f}'.format(df.groupby('subtype').correct.mean().mean()))
print('\n')
