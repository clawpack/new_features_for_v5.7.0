
import os,sys,glob,re

from convert_notebooks import make_html

run_notebooks = True   # if False, only fix existing html files

if 0:
    notebook_files = glob.glob('*.ipynb')
    all_notebooks = [os.path.splitext(f)[0] for f in notebook_files]

if 0:
    all_notebooks = ['Index', 
        'pcolorcells', 
        'NewFeatures', 
        'MakeFlagregionsCoast', 
        'ForceDry', 
        'FGmaxGrids', 
        'FlagRegions', 
        'MarchingFront', 
        'RuledRectangles', 
        'SetEtaInit']

if 1:
    all_notebooks = ['Index']
    
make_html(all_notebooks, run_notebooks=run_notebooks)
