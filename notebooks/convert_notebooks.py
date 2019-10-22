
import os,sys,glob,re

    
def make_html(all_notebooks='all', run_notebooks=True):

    if all_notebooks == 'all':
        notebook_files = glob.glob('*.ipynb')
        all_notebooks = [os.path.splitext(f)[0] for f in notebook_files]
    
    notebook_files = [f+'.ipynb' for f in all_notebooks]

    print('notebook_files = ', notebook_files)
    print('all_notebooks = ', all_notebooks)

    if run_notebooks:
        for file in notebook_files:
            fname = os.path.splitext(file)[0]
            print('converting %s' % file)
            cmd = 'jupyter nbconvert --to html  --execute ' + \
                  '--ExecutePreprocessor.kernel_name=python3 ' +\
                  '--ExecutePreprocessor.timeout=-1  %s' % file
            print(cmd)
            os.system(cmd)

    #html_files = glob.glob('*.html')
    #html_files = [os.path.splitext(file)[0] + '.html' for file in notebook_files]


    html_files = [fname + '.html' for fname in all_notebooks]
    print('html_files = ', html_files)

    for file in html_files:
        infile = open(file,'r')
        lines = infile.readlines()
        infile.close()
        print('Fixing %s' % file)
        with open(file,'w') as outfile:
            for line in lines:
                #for notebook_name in all_notebooks:
                #    line = re.sub(notebook_name+'.ipynb', notebook_name+'.html', line)
                line = re.sub('.ipynb', '.html', line)
                outfile.write(line)


if __name__ == '__main__':
    
    make_html('all')
