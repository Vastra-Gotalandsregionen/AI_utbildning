import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pdb

def show_grid(img,labels=None,figno=1,save=False):
    
    ncol = int(np.round(np.sqrt(img.shape[0])*1.5))
    nrow = int(np.ceil(img.shape[0]/ncol))
    scale = 2
    fig = plt.figure(num=figno,figsize=(ncol*scale, nrow*scale), clear=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrow, ncol),  # creates grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    
    for i in range(img.shape[0]):
        grid[i].imshow(img[i,0,:,:].squeeze(), vmin=0, vmax=0.25, cmap=plt.get_cmap('gray_r'))
        if labels is not None:
            grid[i].text(4, img.size(2)*0.98, '{}'.format(labels[i]),color='red',fontsize=11 )
        grid[i].patch.set_edgecolor('black')  
        grid[i].patch.set_linewidth(1)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    
    for j in range(i+1,nrow*ncol):
        grid[j].remove()
    
    plt.subplots_adjust(wspace=.0,hspace=.0,bottom=0.01,top=0.98,left=0.01,right=0.99)
    plt.show(block=False); plt.draw()
    
    if save:
      plt.savefig('bild_PETbatch.png', dpi=300, facecolor='w', edgecolor=None, format='png', bbox_inches='tight')
    
