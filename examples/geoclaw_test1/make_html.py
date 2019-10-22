
import sys
sys.path.insert(0,'../../notebooks')
import convert_notebooks

all_notebooks = 'all'
run_notebooks = True

convert_notebooks.make_html(all_notebooks,run_notebooks)

