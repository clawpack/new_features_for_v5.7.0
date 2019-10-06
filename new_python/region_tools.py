"""
Tools for defining and working with ruled rectangles.
"""

import numpy as np

class RuledRectangle(object):
    
    def __init__(self, fname=None, slu=None, rect=None):
        self.ixy = None  # 1 or 'x' if s=x,   2 or 'y' if s=y
        self.s = None    # array
        self.lower = None  # array of same length as s
        self.upper = None  # array of same length as s
        self.method = 0  # 0 for pw constant, 1 for pw linear
        self.ds = -1   # > 0 if s values are equally spaced
        if fname is not None:
            self.read(fname)
            if slu is not None:
                print('*** Warning: ignoring slu since fname also given')
            if rect is not None:
                print('*** Warning: ignoring rect since fname also given')
        elif slu is not None:
            self.s = slu[:,0]
            self.lower = slu[:,1]
            self.upper = slu[:,2]
            if rect is not None:
                print('*** Warning: ignoring rect since slu also given')
        elif rect is not None:
            # define a simple rectangle:
            x1,x2,y1,y2 = rect
            self.s = np.array([x1,x2])
            self.lower = np.array([y1,y1])
            self.upper = np.array([y2,y2])
            self.ixy = 1
            self.method = 0
            self.ds = x2 - x1
        
    def bounding_box(self):
        if self.ixy in [1,'x']:
            x1 = self.s.min()
            x2 = self.s.max()
            y1 = self.lower.min()
            y2 = self.upper.max()
        else:
            y1 = self.s.min()
            y2 = self.s.max()
            x1 = self.lower.min()
            x2 = self.upper.max()
        return [x1, x2, y1, y2]

    def slu(self):
        """
        Return npts by 3 array with columns (s, lower, upper)
        """
        slu = np.vstack((self.s, self.lower, self.upper)).T
        return slu

            
    def vertices(self):
        if self.method == 0:
            ss = [self.s[0]]
            for sk in self.s[1:-1]:
                ss += [sk,sk]
            ss = np.array(ss + [self.s[-1]])
            ss = np.hstack((ss, np.flipud(ss), self.s[0]))
            ll = []
            uu = []
            for k in range(len(self.s)-1):
                ll += [self.lower[k],self.lower[k]]
                uu += [self.upper[k],self.upper[k]]
            lu = np.hstack((ll, np.flipud(uu), self.lower[0]))
            
        elif self.method == 1:
            ss = np.hstack((self.s, np.flipud(self.s), self.s[0]))
            lu = np.hstack((self.lower, np.flipud(self.upper), self.lower[0]))
            
        if self.ixy in [1,'x']:
            x = ss
            y = lu
        else:
            x = lu
            y = ss
        return x,y
            
            
    def mask_outside(self, X, Y):
        """
        Given 2d arrays X,Y, return a mask with the same shape with
        mask == True at points that are outside this RuledRectangle.
        So if Z is a data array defined at points X,Y then 
            ma.masked_array(Z, mask) 
        will be a masked array that can be used to plot only the values
        inside the Ruled Region.
        
        Only implemented for self.method == 1 ?
        
        """
        
        transpose_arrays =  (X[0,0] == X[0,-1])
        if transpose_arrays:
            x = X[:,0]
            y = Y[0,:]
        else:
            x = X[0,:]
            y = Y[:,0]
        assert x[0] != x[-1], '*** Wrong orientation?'            
            
        mask = np.empty((len(y),len(x)), dtype=bool)
        mask[:,:] = True
        if self.ixy in [1,'x']:
            iin, = np.where(np.logical_and(self.s.min() < x, x < self.s.max()))
            for i in iin:
                xi = x[i]
                i1, = np.where(self.s < xi)
                if len(i1) > 0:
                    i1 = i1.max()
                    if i1 < len(self.s)-1:
                        alpha = (xi-self.s[i1])/(self.s[i1+1]-self.s[i1])
                        ylower = (1-alpha)*self.lower[i1] + \
                                     alpha*self.lower[i1+1]
                        yupper = (1-alpha)*self.upper[i1] + \
                                     alpha*self.upper[i1+1]   
                        j, = np.where(np.logical_and(ylower < y, y < yupper))
                        mask[j,i] = False
        elif self.ixy in [2,'y']:
            jin, = np.where(np.logical_and(self.s.min() < y, y < self.s.max()))
            for j in jin:
                yj = y[j]
                i1, = np.where(self.s < yj)
                if len(i1) > 0:
                    i1 = i1.max()
                    if i1 < len(self.s)-1:
                        alpha = (yj-self.s[i1])/(self.s[i1+1]-self.s[i1])
                        xlower = (1-alpha)*self.lower[i1] + \
                                     alpha*self.lower[i1+1]
                        xupper = (1-alpha)*self.upper[i1] + \
                                     alpha*self.upper[i1+1]   
                        i, = np.where(np.logical_and(xlower < x, x < xupper))
                        mask[j,i] = False
        if transpose_arrays:
            mask = mask.T
            
        return mask
            
        
    def write(self, fname):
        slu = self.slu()
        ds = self.s[1:] - self.s[:-1]
        dss = ds.max() - ds.min()
        if dss < 1e-6*ds.max():
            self.ds = ds.max()
        else:
            self.ds = -1  # not uniformly spaced
            
        # if ixy is 'x' or 'y' replace by 1 or 2 for writing:
        if self.ixy in [1,'x']:
            ixyint = 1
        else:
            ixyint = 2
            
        header = """\n%i   ixy\n%i   method\n%g    ds\n%i    nrules""" \
            % (ixyint,self.method,self.ds,len(self.s))
        np.savetxt(fname, slu,header=header,comments='',fmt='%.9f  ')

    def read(self, fname):
        lines = open(fname,'r').readlines()
        k = -1
        comments = True
        while comments:
            k += 1
            line = lines[k].strip()
            if (line != '') and (line[0] != '#'):
                comments = False
        self.ixy = int(line.split()[0])
        k += 1
        self.method = int(lines[k].split()[0])
        k += 1
        self.ds = float(lines[k].split()[0])
        k += 1
        self.nrules = int(lines[k].split()[0])
        slu = np.loadtxt(fname, skiprows=k+1)
        assert slu.shape[0] == self.nrules, '*** wrong shape'
        self.s = slu[:,0]
        self.lower = slu[:,1]
        self.upper = slu[:,2]
        
    def make_kml(self, fname='RuledRectangle.kml', name='RuledRectangle', 
                 color='00FFFF', width=2, verbose=False):
        from clawpack.geoclaw import kmltools
        x,y = self.vertices()
        kmltools.poly2kml((x,y), fname=fname, name=name, color=color, 
                          width=width, verbose=verbose)
        
        
def ruledrectangle_covering_selected_points(X, Y, pts_chosen, ixy, method=0,
                                            padding=0, verbose=True):

    if np.ndim(X) == 2:
        x = X[0,:]
        y = Y[:,0]
    else:
        x = X
        y = Y

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if ixy in [1,'x']:

        # Ruled rectangle with s = x:

        s = []
        lower = []
        upper = []
        for i in range(len(x)):
            if pts_chosen[:,i].sum() > 0:
                j = np.where(pts_chosen[:,i]==1)[0]
                j1 = j.min()
                j2 = j.max()
                s.append(x[i])
                lower.append(y[j1])
                upper.append(y[j2])
                
    elif ixy in [2,'y']:

        # Ruled rectangle with s = y:

        s = []
        lower = []
        upper = []

        for j in range(len(y)):
            if pts_chosen[j,:].sum() > 0:
                i = np.where(pts_chosen[j,:]==1)[0]
                i1 = i.min()
                i2 = i.max()
                s.append(y[j])
                lower.append(x[i1])
                upper.append(x[i2])
                
    else:
        raise(ValueError('Unrecognized value of ixy'))

    s = np.array(s)
    lower = np.array(lower)
    upper = np.array(upper)
    ds = s[1] - s[0]

    if method == 0:
        # extend so rectangles cover grid cells with centers at (x,y)
        if verbose:
            print('Extending rectangles to cover grid cells')
        if abs(dx - dy) > 1e-6:
            print('*** Warning, dx = %.8e not equal to dy = %.8e' \
                  % (dx, dy))
        lower = lower - 0.5*ds
        upper = upper + 0.5*ds
        s = s - 0.5*ds
        s = np.hstack((s, s[-1]+ds))
        lower = np.hstack((lower, lower[-1]))
        upper = np.hstack((upper, upper[-1]))
        
    rr = RuledRectangle()
    rr.ixy = ixy
    rr.s = s
    rr.lower = lower
    rr.upper = upper
    rr.method = method
    rr.ds = ds

    if verbose:
        rr_npts = int(np.ceil(np.sum(rr.upper - rr.lower) / ds))
        print('RuledRectangle covers %s grid points' % rr_npts)

    return rr
