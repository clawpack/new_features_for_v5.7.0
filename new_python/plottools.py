
"""
Module with some miscellaneous useful tools.
"""

from __future__ import absolute_import
def plotbox(xy, kwargs={'color':'b', 'linewidth':2}):
    """
    Add a box around a region to an existing plot.
    xy can be a list of [x1, x2, y1, y2] of the corners 
    or a string "x1 x2 y1 y2".
    """
    from pylab import plot
    if type(xy)==str:
        xy=xy.split()
    x1 = float(xy[0])
    x2 = float(xy[1])
    y1 = float(xy[2])
    y2 = float(xy[3])
    plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],**kwargs)

def pcolorcells(X, Y, Z, ax=None, **kwargs):
    """
    Wraps pcolormesh in a way that works if X,Y are cell centers or edges.
    X,Y can be 2d or 1d arrays. 
    
    if `ax==None` then the plot is done on a new set of axes, otherwise on ax

    X,Y,Z is the data to be plotted.  It is assumed to be finite volume data
    where Z[i,j] is a constant value over a grid cell.

    Internally x,y are defined as 1d arrays since it is assumed the 
    grids are Cartesian.
    
    If the length of the 1d arrays x and y match the dimensions of Z then
    these are assumed to be cell center values. In this case the arrays
    are expanded by one to obtain x_edge, y_edge as edge values,
    as needed for proper alignment.

    If the length of x,y is already one greater than the corresponding
    dimension of Z, then it is assumed that these are already edge values.
    
    Notes: 
    
    - This should work also if x and/or y is decreasing rather than increasing.  
    
    - Currently assumes x,y are 1d arrays, could extend to also allow 2d
      arrays as input.
      
    """
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    # If X is 2d extract proper 1d slice:
    if X.ndim == 1:
        x = X
    elif X.ndim == 2:
        if X[0,0] == X[0,1]:
            x = X[:,0]
        else:
            x = X[0,:]
            
    # If Y is 2d extract proper 1d slice:
    if Y.ndim == 1:
        y = Y
    elif Y.ndim == 2:
        if Y[0,0] == Y[0,1]:
            y = Y[:,0]
        else:
            y = Y[0,:]                    

    #dx = x[1]-x[0]
    #dy = y[1]-y[0]
    
    diffx = np.diff(x)
    diffy = np.diff(y)
    dx = np.mean(diffx)
    dy = np.mean(diffy)
    
    if diffx.max()-diffx.min() > 1e-3*dx:
        raise ValueError("x must be equally spaced for pcolorcells")
    if diffy.max()-diffy.min() > 1e-3*dy:
        raise ValueError("y must be equally spaced for pcolorcells")


    if len(x) == Z.shape[1]:
        # cell centers, so xedge should be expanded by dx/2 on each end:
        xedge = np.arange(x[0]-0.5*dx, x[-1]+dx, dx)
    elif len(x) == Z.shape[1]+1:
        # assume x already contains edge values
        xedge = x
    else:
        raise ValueError('x has unexpected length')

    if len(y) == Z.shape[0]:
        # cell centers, so xedge should be expanded by dx/2 on each end:
        yedge = np.arange(y[0]-0.5*dy, y[-1]+dy, dy)
    elif len(y) == Z.shape[0]+1:
        # assume x already contains edge values
        yedge = y
    else:
        raise ValueError('y has unexpected length')
        
    if ax is None:
        pc = plt.pcolormesh(xedge, yedge, Z, **kwargs)
    else:
        pc = ax.pcolormesh(xedge, yedge, Z, **kwargs)
    
    return pc
