# import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

def plot(save=False):
    ## Data
    df = pd.read_csv('results/convergence.csv')
    
    ### Plot
    sns.set(style="ticks")
    sns.set_style("darkgrid")
    
    g = sns.FacetGrid(df, col="metric",hue='split',  margin_titles=True, sharey=False, height=6)
    g.map(sns.lineplot, "epoch", "value")
    plt.legend()
    
    plt.show(block=False); plt.draw() 
    if save:
      plt.savefig('bild_konvergens.png', dpi=300, facecolor='w', edgecolor=None, format='png', bbox_inches='tight')
