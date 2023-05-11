import numpy as np
import pandas as pd
import torch

####################################################
### Dataset-klass som anropas av dataloader
class my_dataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe,transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform
    
    def __getitem__(self, index):
        # Plocka ut ett index från stora datastrukturen (en patient/bild)
        dataframe = self.dataframe.iloc[index]
        # Läs etikett (label) och patient ID från datastrukturen
        label     = dataframe.label
        subjectID = dataframe.Subject_ID
        # Läs patientens bild från disk, storlek [C,H,W]
        img       = get_image(subjectID,self.transform)
        
        return img, label
    
    def __len__(self):
        return len(self.dataframe)

####################################################
### Funktion som läser bilddata från disk
def get_image(subjectID,transform=None):
    
    # Identifiera vilken bild som skall läsas
    filename='/home/ida/projekt/Rontgenveckan_2022/data/PET_binaries/{}_cor.bin'.format(subjectID)
    
    # Läs binär fil från disk
    img  = np.fromfile(filename, dtype='float32')
    
    # Forma om vektor till kvadratisk bild [H,W]
    img  = np.reshape( img,[310,310] )
    
    # Normalisera bild till [0,1]
    img_min, img_max = 0,30
    img = img.clip(img_min,img_max)
    img = (img-img_min)/(img_max-img_min)
    
    # Konvertera från numpy till tensor. Addera en första dimension att representera kanal C: [C,H,W]
    img = torch.FloatTensor(img).unsqueeze(0)
    
    # Transformation i farten för augmentering
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    
    return img

####################################################
### Funktion som slumpvis skalar om bild
class RandomScale(object):
    """Randomly scale the 2D image.
    """
    def __call__(self, image):
        scale = np.random.uniform(low=0.9, high=1.1, size=1)
        image = image*scale[0]
        return image

