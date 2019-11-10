
import sys
sys.path.insert(0,'../../notebooks')
import convert_notebooks

all_notebooks = ['geoclaw_test1_index.ipynb',
                 'MakeInputFiles_test1.ipynb',
                 'RunGeoclaw_test1.ipynb']
run_notebooks = True

convert_notebooks.make_html(all_notebooks,run_notebooks)

