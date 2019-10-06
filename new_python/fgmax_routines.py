"""
Create fgmax input grids and process fgmax output from GeoClaw run.

"""
from __future__ import print_function

import sys

if 'matplotlib' not in sys.modules:
    # if not running interactively...
    import matplotlib
    matplotlib.use('Agg')  # Use an image backend

import numpy
import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import ma
from clawpack.geoclaw import topotools
from clawpack.geoclaw.dtopotools import DTopography
from clawpack.geoclaw.kmltools import box2kml
import fgmax_tools  # local version
import matplotlib.colors as colors
import clawpack.visclaw.colormaps as colormaps

import csv


try:
    rootdir = os.environ['WA_EMD_2019']
except:
    print("*** Need to set environment variable WA_EMD_2019")
    sys.exit()

#print('in fgmax_routines, rootdir = ',rootdir)

# gauge locations:
#gauges_csv_file = rootdir + '/info/tide_gauge_locations.csv'
gauges_csv_file = os.path.join(rootdir, 'info/UW_2019_gauge_locations.csv')
#print('in fgmax_routines, gauges_csv_file = ',gauges_csv_file)


f = open(gauges_csv_file,'r')
csv_reader = csv.reader(f)

gaugeloc = {}
gaugexy = {}
gauge_domain = {}
gaugenos = []
for row in csv_reader:
    try:
        xg = float(row[2])
        yg = float(row[3])
    except:
        continue  # go to next row
    gaugeno = int(row[0])
    #if gaugeno<100:
        #gaugeno += 100
    gaugeloc[gaugeno] = row[4]
    gauge_domain[gaugeno] = row[5]
    gaugexy[gaugeno] = (xg,yg)
    gaugenos.append(gaugeno)



def make_fgmax_new(params):
    import matplotlib.pyplot as plt
    
    arcsec13 = 1./(3*3600.)  # 1/3 arc second
    arcsec16 = 1./(6*3600.)  # 1/6 arc second

    print("Location: %s" % params.loc)
    loc = params.loc
    
    # expand the desired area a bit for overlap:
    x1 = params.x1_desired - 3*arcsec13
    x2 = params.x2_desired + 3*arcsec13
    y1 = params.y1_desired - 3*arcsec13
    y2 = params.y2_desired + 3*arcsec13
    extent = [x1, x2, y1, y2]

    kml_fname = 'fgmax_%s.kml' % loc
    kml_name = 'fgmax_%s' % loc
    box2kml(extent, fname=kml_fname, name=kml_name, color='FFFF00', 
            width=2, verbose=True)
    
    topofile = params.topo_fname
    topo_type = 3
    dx = dy = 1./(3*3600.)

    zmin,zmax = -10000,40   # select only point for which zmin <= z <= zmax


    fg, masked_topo = fgmax_tools.make_fgmax_points_from_topo(topofile,topo_type,
                        zmin,zmax,extent)

    # fg is returned as an object of class geoclaw.fgmax_tools.FGmaxGrid with:
    #   fg.point_style = 0,
    #   fg.X, fg.Y  set to a list of points
    #   fg.Z is Z values
    #   fg.npts = len(fg.X)

    # The header will be created separately for each earthquake source
    # Here we just output the data values:
    xyz = numpy.vstack([fg.X, fg.Y, fg.Z]).T
    fname = 'fgmax_%s_points_xyz.data' % loc
    numpy.savetxt(fname, xyz, header='%8i' % fg.npts, comments='', fmt='%24.14e')
    print('Created ',fname)

    if params.make_plot:
        png_fname = '%s_fgmax_points.png' % loc
        fig = plt.figure(500,figsize=params.figsize)
        plt.clf()
        ax = plt.subplot(111)
        title = '%s\n%i fgmax points' % (params.loc,masked_topo.Z.count())
        # fgmax_tools.plot_fgmax_points(fg.X,fg.Y,fg.Z,dx,dy,-40,zmax,
        #                       ax=ax,png_fname=png_fname)
        #fgmax_tools.plot_fgmax_masked_topo(masked_topo,zmin=-40,zmax=40,
        #                      ax=ax,title=title,png_fname=png_fname)
                                  
        print("Plotting %i fgmax_points... " % masked_topo.Z.count())
        
        zmin = masked_topo.Z.min()
    
        zmax = masked_topo.Z.max()
        
        land_cmap = colormaps.make_colormap({ 0.0:[0.1,0.4,0.0],
                                             0.25:[0.0,1.0,0.0],
                                              0.5:[0.8,1.0,0.5],
                                              1.0:[0.8,0.5,0.2]})

        sea_cmap = colormaps.make_colormap({ 0.0:[0,0,1], 1.:[.8,.8,1]})

        cmap, norm = colormaps.add_colormaps((land_cmap, sea_cmap),
                                             data_limits=(zmin,zmax),
                                             data_break=0.)
        
        pc = plt.pcolor(masked_topo.X, masked_topo.Y, masked_topo.Z,
                        cmap=cmap, norm=norm)
        cb = plt.colorbar(pc, extend='both')
        cb.set_label('meters')

        plt.gca().set_aspect(1./numpy.cos(masked_topo.Y.mean()*numpy.pi/180))
        
        x1 = masked_topo.X.min() - 10*(masked_topo.x[1]-masked_topo.x[0])
        x2 = masked_topo.X.max() + 10*(masked_topo.x[1]-masked_topo.x[0])
        y1 = masked_topo.Y.min() - 10*(masked_topo.y[1]-masked_topo.y[0])
        y2 = masked_topo.Y.max() + 10*(masked_topo.y[1]-masked_topo.y[0])
        plt.axis([x1,x2,y1,y2])
        
        plt.ticklabel_format(format='plain',useOffset=False)
        plt.xticks(rotation=20)
        plt.title(title)

        if 1:
            # plot gauges in this region:
            for gaugeno in gaugenos:
                xg,yg = gaugexy[gaugeno]
                #print('gauge ',gaugeno, xg, yg)
                if (gaugeno<300) and (x1 < xg < x2) and (y1 < yg < y2):
                    print('    Adding gauge %s' % gaugeno)
                    plt.plot([xg],[yg],'kx')
                    plt.text(xg+.002,yg-.001,gaugeno,color='k', fontsize=12)
                    
        if png_fname is not None:
            plt.savefig(png_fname)
            print("Created %s" % png_fname)
            plt.close(500)
    if 0:
        # currently done in topo/make_fgmax_points.py
        fname = 'fgmax_%s_maskedtopo.nc' % fgmax_name
        topom.write(fname, topo_type=4, fill_value=-9999.)
        print('Created ',fname)
    
    return fg, masked_topo

def process_fgmax(params):
    from scipy.interpolate import RegularGridInterpolator

    loc = params.loc
    event = params.event

    print('loc = %s, event = %s' % (loc,event))
    print('should agree with this directory: %s' % os.getcwd())

    # Read fgmax data:
    fg = fgmax_tools.FGmaxGrid()
    fgmax_input_file_name = 'fgmax_%s' % loc
    print('fgmax input file: %s' % fgmax_input_file_name)
    fg.read_input_data(fgmax_input_file_name + '_header.data')

    print('reading fgmax output from %s' % params.outdir)
    fg.read_output(outdir=params.outdir)
    xx = fg.X
    yy = fg.Y
    #zz = fg.Z  # not available from fgmax_pts_topostyle file


    # compute subsidence/uplift at each fgmax point:
    
    if params.variable_sea_level:
        # compute subsidence:
        dtopo = DTopography()
        dtopo.read(params.dtopo_path, dtopo_type=3)
        x1d = dtopo.X[0,:]
        y1d = dtopo.Y[:,0]
        dtopo_func = RegularGridInterpolator((x1d,y1d), dtopo.dZ[-1,:,:].T, 
                        method='linear', bounds_error=False, fill_value=0.)

            
    # convert to masked array on uniform grid for .nc file and plots:
    
    fgm = fgmax_tools.FGmaxMaskedGrid()
    dx = dy = 1./(3*3600.)  # For 1/3 arcsecond fgmax grid
    
    # convert to arrays and create fgm.X etc.
    #   (need a version to put on existing X,Y grid?)
    fgm.convert_lists_to_arrays(fg,dx,dy) 
    fgm.x = fgm.X[0,:]
    fgm.y = fgm.Y[:,0]
    
    if params.variable_sea_level: 
        dz = dtopo_func(list(zip(numpy.ravel(fgm.X), numpy.ravel(fgm.Y))))
        fgm.dz = numpy.reshape(dz, fgm.X.shape)
    else:
        fgm.dz = numpy.zeros(fgm.X.shape)
            
    fgm.B0 = fgm.B - fgm.dz   # original topo before subsidence/uplift
    
    use_wet_mask = params.use_wet_mask
    
    if params.use_wet_mask:
        fname_wet_mask = params.fname_wet_mask
        print('Reading wet_mask from ',fname_wet_mask)
        wet_mask = topotools.Topography()
        wet_mask.read(fname_wet_mask, topo_type=3)
        i1 = int(round((fgm.x[0]-wet_mask.x[0])/dx))
        i2 = int(round((fgm.x[-1]-wet_mask.x[0])/dx))
        j1 = int(round((fgm.y[0]-wet_mask.y[0])/dy))
        j2 = int(round((fgm.y[-1]-wet_mask.y[0])/dy))
        if (i1<0) or (i2-i1+1 != len(fgm.x)) or \
           (j1<0) or (j2-j1+1 != len(fgm.y)):
            print('*** wet_mask does not cover fgm extent, not using')
            use_wet_mask = False
            fgm.allow_wet_init = None
        else:
            fgm.allow_wet_init = wet_mask.Z[j1:j2+1, i1:i2+1]
    else:
        fgm.allow_wet_init = None
        print('*** use_wet_mask is False')
        
    if fgm.allow_wet_init is not None:
        fgm.h_onshore = ma.masked_where(fgm.allow_wet_init==1., fgm.h)
    else:
        fgm.h_onshore = ma.masked_where(fgm.B0 < 0., fgm.h)
    
    return fgm
    
def plot_fgmax_masked(params, fgm):

    figsize = params.figsize
    plot_shore = params.plot_shore
    if plot_shore:
        shore_x = params.shorelines[:,0]
        shore_y = params.shorelines[:,1]

    #units used in Geoclaw:
    #  'depth':'meters', 'speed':'m/s', 'mflux':'m**3/s**2'
      
    # plot units can be:
    #   'depth': 'meters' or 'feet'
    #   'speed': 'm/s' or 'knots'
    #   'mflux': 'm**3/s**2' or 'kN/m'
    #   'atime': 'seconds' or 'minutes' or 'hours'
    plot_units = {'depth':'meters', 'speed':'knots', \
                  'mflux':'kN/m', 'atime':'minutes'}

    # conversion factors:
    depth_to_feet = 1./0.3048  # exact value for meters to feet
    speed_to_knots = 1.94384 # m/s to knots
    atime_to_minutes = 1./60.
    atime_to_hours = 1./3600.

    mflux_to_kNm = 1.8 # m**3/s**2 to kN/m
      # based on:
      # drag coefficient C_d = 2,  
      # and density of sediment-laden water is rho = 1200 kg/m**3.
      # as suggested in FEMA P646
      #  http://www.fema.gov/media-library-data/ 20130726-1641-20490-9063/femap646.pdf, 2008.


    # -------------------
    # Depth plot:
    # -------------------

    if plot_units['depth'] == 'meters':
        depth_factor = 1.
        bounds = numpy.array([1e-6,0.25,0.5,0.75,1,1.25,1.5])
    elif plot_units['depth'] == 'feet':
        depth_factor = depth_to_feet
        bounds = numpy.array([1e-6,1,2.5,4,6,9,12]) 
    else:
        raise Exception("*** unrecognized plot_units['depth']")

    
    h = fgm.h * depth_factor
    h_onshore = fgm.h_onshore * depth_factor
    
    #hdry = ma.masked_where(h > 1e-6, h_onshore)
    #hwet = ma.masked_where(h < 1e-6, h_onshore)
                     
    cmap = mpl.colors.ListedColormap([[.7,.7,1],\
                     [.5,.5,1],[0,0,1],\
                     [1,.7,.7], [1,.4,.4], [1,0,0]])
                     
    # Set color for value exceeding top of range to purple:
    cmap.set_over(color=[1,0,1])
    
    # Set color for land points without inundation to light green:
    cmap.set_under(color=[.7,1,.7])

    #imshow(GEmap,extent=GEextent)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    cb_label = 'depth (%s)' % plot_units['depth']
    title = '%s\nDepth at fgmax points' % params.loc

    png_fname = '%s_depth.png' % params.loc

    plt.figure(100, figsize=figsize)
    plt.clf()
    ax1 = plt.axes()

    pc = plt.pcolormesh(fgm.X,fgm.Y, h_onshore, cmap=cmap,norm=norm)    
    
    cb = plt.colorbar(pc, extend='max')
    if cb_label is not None:
        cb.set_label(cb_label)

    plt.ticklabel_format(format='plain',useOffset=False)
    plt.xticks(rotation=20)
    
    if plot_shore: 
        if use_wet_mask:
            plt.contour(fgm.X,fgm.Y,fgm.allow_wet_init,[0.5],colors='g',lw=0.5)
        else:
            plt.plot(shore_x, shore_y, 'g', lw=0.5)

    # -------------------
    # Speed plot:
    # -------------------
    if plot_units['speed'] == 'm/s':
        speed = fgm.s
        bounds = numpy.array([1e-6,0.5,1.5,2,2.5,3,4.5,6])
    elif plot_units['speed'] == 'knots':
        speed = fgm.s * speed_to_knots
        bounds = numpy.array([1e-6,1,3,4,5,6,9,12])
    else:
        raise Exception("*** unrecognized plot_units['speed']")

    if 0:
        # to better see small speeds when testing SFS:
        plot_units['speed'] = 'm/s'
        speed = fgm.s
        bounds = numpy.array([1e-6,0.1,0.25,0.5,0.75,1.,1.5,2.])

    cmap = mpl.colors.ListedColormap([[.9,.9,1],[.6,.6,1],\
                     [.3,.3,1],[0,0,1],\
                     [1,.7,.7], [1,.4,.4], [1,0,0]])
                     
    # Set color for value exceeding top of range to purple:
    cmap.set_over(color=[1,0,1])
    
    # Set color for land points without inundation to light green:
    cmap.set_under(color=[.7,1,.7])
    
    #imshow(GEmap,extent=GEextent)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    cb_label = 'speed (%s)' % plot_units['speed']
    title = '%s\nSpeed at fgmax points' % params.loc
    png_fname = '%s_speed.png' % params.loc

    plt.figure(101, figsize=figsize)
    plt.clf()
    ax1 = plt.axes()

    pc = plt.pcolormesh(fgm.X,fgm.Y, speed, cmap=cmap,norm=norm)
    
    cb = plt.colorbar(pc, extend='max')
    if cb_label is not None:
        cb.set_label(cb_label)

    plt.ticklabel_format(format='plain',useOffset=False)
    plt.xticks(rotation=20)
    
    if plot_shore: 
        if use_wet_mask:
            plt.contour(fgm.X,fgm.Y,fgm.allow_wet_init,[0.5],colors='g',lw=0.5)
        else:
            plt.plot(shore_x, shore_y, 'g', lw=0.5)


def make_nc_input(fname_nc, fgm, force=False, verbose=True):

    import netCDF4
    import time
    import os
    
    if os.path.isfile(fname_nc):
        if force and verbose:
            print('Overwriting ', fname_nc)
        elif not force:
            print('*** netCDF file already exists, \n'\
                + '*** NOT overwriting '\
                + '--- use force==True to overwrite' )
            return
    
    with netCDF4.Dataset(fname_nc, 'w') as rootgrp:

        rootgrp.description = "fgmax data for " + fgm.id
        rootgrp.history = "Created with input data " + time.ctime(time.time())
        rootgrp.history += " in %s;  " % os.getcwd()
            
        if fgm.X is not None:
            x = fgm.X[0,:]
            lon = rootgrp.createDimension('lon', len(x))
            longitudes = rootgrp.createVariable('lon','f8',('lon',))
            longitudes[:] = x
            longitudes.units = 'degrees_east'
        else:
            if verbose: print('fgm.X is None, not adding x')
            
        if fgm.Y is not None:
            y = fgm.Y[:,0]
            lat = rootgrp.createDimension('lat', len(y))
            latitudes = rootgrp.createVariable('lat','f8',('lat',))
            latitudes[:] = y
            latitudes.units = 'degrees_north'
        else:
            if verbose: print('fgm.Y is None, not adding y')
            
        if fgm.Z is not None:
            Z = rootgrp.createVariable('Z','f4',('lat','lon',))
            Z[:,:] = fgm.Z.data  # include points that are not fgmax_points
            Z.units = 'meters'
        else:
            if verbose: print('fgm.Z is None, not adding')
            
        if fgm.fgmax_point is not None:
            fgmax_point_var = \
                rootgrp.createVariable('fgmax_point','u1',('lat','lon',))
            fgmax_point_var[:,:] = fgm.fgmax_point
        else:
            if verbose: print('fgm.fgmax_point is None, not adding')
            
        if fgm.allow_wet_init is not None:
            allow_wet_init = \
                rootgrp.createVariable('allow_wet_init','u1',('lat','lon',))
            allow_wet_init[:,:] = fgm.allow_wet_init
        else:
            if verbose: print('fgm.allow_wet_init is None, not adding')  

        print('Created %s' % fname_nc)            
        if verbose:
            print('History:  ', rootgrp.history)      
        
def write_nc_output(fname_nc, fgm, new=False, force=False, 
                    outdir='Unknown', verbose=True):

    import netCDF4
    import time
    import os
    from clawpack.clawutil.data import ClawData 
    
    fv = -9999.   # fill_value for netcdf4
    
    if new:
        # first create a new .nc file with X,Y,fgmax_point,allow_wet_init:
        make_nc_input(fname_nc, fgm, force=force, verbose=verbose)        
        
    if outdir is 'Unknown':
        # Cannot determine tfinal or run_finished time
        tfinal = fv
        run_finished = 'Unknown'
    else:
        claw = ClawData()
        claw.read(outdir+'/claw.data', force=True)
        tfinal = claw.tfinal
        
        try:
            mtime = os.path.getmtime(outdir+'/timing.txt')
            run_finished = time.ctime(mtime) 
        except:
            run_finished = 'Unknown'
            
    # add fgmax output results to existing file
    with netCDF4.Dataset(fname_nc, 'a') as rootgrp:
        if verbose:
            print('Appending data from fgm to nc file',fname_nc)
            print('        nc file description: ', rootgrp.description)
            print('        fgm.id: ', fgm.id)
        
        h = rootgrp.variables.get('h', None)
        if (h is not None) and (not force):
            print('*** netCDF file already contains output,\n'\
                + '*** NOT overwriting '\
                + '--- use force==True to overwrite' )
            return
                
        x = numpy.array(rootgrp.variables['lon'])
        y = numpy.array(rootgrp.variables['lat'])
        X,Y = numpy.meshgrid(x,y)
        Z = numpy.array(rootgrp.variables['Z'])
        fgmax_point = numpy.array(rootgrp.variables['fgmax_point'])
        bounding_box = [x.min(),x.max(),y.min(),y.max()]
        
        dx = x[1]-x[0]
        Xclose = numpy.allclose(fgm.X, X, atol=0.1*dx)
        Yclose = numpy.allclose(fgm.Y, Y, atol=0.1*dx)
        
        if (fgm.X.shape != X.shape) or (not Xclose) or (not Yclose):
            # for now raise an exception, might want to extent to allow
            # filling only part of input arrays
            print('*** Mismatch of fgm with data in nc file:')
            print('fgm.X.shape = ',fgm.X.shape)
            print('nc  X.shape = ',X.shape)
            print('fgm.bounding_box = ',fgm.bounding_box())
            print('nc  bounding_box = ',bounding_box)
            raise ValueError('*** Mismatch of fgm with data in nc file')
    

        rootgrp.history += "Added output " + time.ctime(time.time())
        rootgrp.history += " in %s;  " % os.getcwd()
        
        rootgrp.tfinal = tfinal
        rootgrp.outdir = os.path.abspath(outdir)
        rootgrp.run_finished = run_finished
        
        fgmax_point = rootgrp.variables.get('fgmax_point', None)

        if fgm.dz is not None:
            try:
                dz = rootgrp.variables['dz']
            except:
                dz = rootgrp.createVariable('dz','f4',('lat','lon',),
                                            fill_value=fv)
            dz[:,:] = fgm.dz
            dz.units = 'meters'
            if verbose: print('    Adding fgm.dz to nc file')
        else:
            if verbose: print('fgm.dz is None, not adding')

        if fgm.B is not None:
            try:
                B = rootgrp.variables['B']
            except:
                B = rootgrp.createVariable('B','f4',('lat','lon',),
                                            fill_value=fv)
            B[:,:] = fgm.B
            B.units = 'meters'
            if verbose: print('    Adding fgm.B to nc file')
        else:
            if verbose: print('fgm.B is None, not adding')
                        
        if fgm.h is not None:
            try:
                h = rootgrp.variables['h']
            except:
                h = rootgrp.createVariable('h','f4',('lat','lon',),
                                            fill_value=fv)
            h[:,:] = fgm.h
            h.units = 'meters'
            if verbose: print('    Adding fgm.h to nc file')
        else:
            if verbose: print('fgm.h is None, not adding')
            
        if fgm.s is not None:        
            try:
                s = rootgrp.variables['s']
            except:
                s = rootgrp.createVariable('s','f4',('lat','lon',),
                                            fill_value=fv)
            s[:,:] = fgm.s
            s.units = 'meters/second'
            if verbose: print('    Adding fgm.s to nc file')
        else:
            if verbose: print('fgm.s is None, not adding')
            
        if fgm.hss is not None:        
            try:
                hss = rootgrp.variables['hss']
            except:
                hss = rootgrp.createVariable('hss','f4',('lat','lon',),
                                            fill_value=fv)
            hss[:,:] = fgm.hss
            hss.units = 'meters^3/sec^2'
            if verbose: print('    Adding fgm.hss to nc file')
        else:
            if verbose: print('fgm.hss is None, not adding')
            
        if fgm.hmin is not None:        
            try:
                hmin = rootgrp.variables['hmin']
            except:
                hmin = rootgrp.createVariable('hmin','f4',('lat','lon',),
                                            fill_value=fv)
            # negate hmin so that it is minimum flow depth min(h):
            hmin[:,:] = -fgm.hmin
            hmin.units = 'meters'
            if verbose: print('    Adding fgm.hmin to nc file')
        else:
            if verbose: print('fgm.hmin is None, not adding')
            
        if fgm.arrival_time is not None:        
            try:
                arrival_time = rootgrp.variables['arrival_time']
            except:
                arrival_time = rootgrp.createVariable('arrival_time','f4',('lat','lon',),
                                            fill_value=fv)
            arrival_time[:,:] = fgm.arrival_time
            arrival_time.units = 'seconds'
            if verbose: print('    Adding fgm.arrival_time to nc file')
        else:
            if verbose: print('fgm.arrival_time is None, not adding')
            
        print('Created %s' % fname_nc)
        if verbose:
            print('History:  ', rootgrp.history)
            print('\nMetadata:')
            print('  outdir:  ', rootgrp.outdir)
            print('  run_finished:  ', rootgrp.run_finished)
            print('  tfinal:  ', rootgrp.tfinal)

def read_nc(fname_nc, verbose=True):

    import netCDF4
    import time
    import os

                
    def get_as_array(var, fgmvar=None):
        if fgmvar is None:
            fgmvar = var
        a = rootgrp.variables.get(var, None)
        if a is not None:
            if verbose: print('    Loaded %s as fgm.%s' % (var,fgmvar))
            return numpy.array(a)
        else:
            if verbose: print('    Did not find %s for fgm.%s' \
                                % (var,fgmvar))
            return None
                    
    fgm = fgmax_tools.FGmaxMaskedGrid()

    with netCDF4.Dataset(fname_nc, 'r') as rootgrp:
        if verbose:
            print('Reading data to fgm from nc file',fname_nc)
            print('        nc file description: ', rootgrp.description)
            print('History:  ', rootgrp.history)


                
        x = get_as_array('lon','x')
        y = get_as_array('lat','y')
        
        if (x is None) or (y is None):
            print('*** Could not create grid')
            return None
            
        X,Y = numpy.meshgrid(x,y)
        fgm.X = X
        fgm.Y = Y
        if verbose:
            print('    Constructed fgm.X and fgm.Y')
        
        fgm.Z = get_as_array('Z')
        fgm.B = get_as_array('B')
        fgm.fgmax_point = get_as_array('fgmax_point') 
        fgm.dz = get_as_array('dz')
        fgm.h = get_as_array('h')
        fgm.s = get_as_array('s')
        fgm.hss = get_as_array('hss')
        fgm.hmin = get_as_array('hmin')
        fgm.arrival_time = get_as_array('arrival_time')
        fgm.allow_wet_init = get_as_array('allow_wet_init')
        
    if verbose:
        print('Returning FGmaxMaskedGrid object fgm')
    return fgm
