
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from clawpack.geoclaw import topotools
from six.moves import range
import os,sys


new_code = '../../new_python'
if 'new_python' not in sys.path[0] + sys.path[1]:
    print('sys.path[0] = ',sys.path[0])
    print('Adding %s to path' % new_code)
    sys.path.insert(0, new_code)


sea_level = 3.1  # lake elevation, should agree with setrun

# adjust these values if needed for different size tsunamis:
ylim_transects = [-5,10]  # y-axis limits for transect plots
cmax_land = 40.  # for color scale of land (greens)
camplitude = 2.  # for color scale on planview plots

# make symmetric about sea_level:
cmin = sea_level-camplitude
cmax = sea_level+camplitude

def surface_or_depth_lake(current_data):
    """
    Return a masked array containing the surface elevation where the topo is
    below sea level or the water depth where the topo is above sea level.
    Mask out dry cells.  Assumes sea level is at topo=0.
    Surface is eta = h+topo, assumed to be output as 4th column of fort.q
    files.

    Modified from visclaw.geoplot version to use sea_level
    """

    drytol = getattr(current_data.user, 'drytol', 1e-3)
    q = current_data.q
    h = q[0,:,:]
    eta = q[3,:,:]
    topo = eta - h

    # With this version, the land is transparent.
    surface_or_depth = np.ma.masked_where(h <= drytol,
                                             np.where(topo<sea_level, eta, h))

    try:
        # Use mask covering coarse regions if it's set:
        m = current_data.mask_coarse
        surface_or_depth = np.ma.masked_where(m, surface_or_depth)
    except:
        pass

    return surface_or_depth



#--------------------------
def setplot(plotdata=None):
#--------------------------

    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of pyclaw.plotters.data.ClawPlotData.
    Output: a modified version of plotdata.

    """


    from clawpack.visclaw import colormaps, geoplot
    from numpy import linspace

    if plotdata is None:
        from clawpack.visclaw.data import ClawPlotData
        plotdata = ClawPlotData()


    plotdata.clearfigures()  # clear any old figures,axes,items data
    plotdata.format = 'binary'



    def timeformat(t):
        from numpy import mod
        hours = int(t/3600.)
        tmin = mod(t,3600.)
        min = int(tmin/60.)
        sec = int(mod(tmin,60.))
        timestr = '%s:%s:%s' % (hours,str(min).zfill(2),str(sec).zfill(2))
        return timestr

    def title_hours(current_data):
        from pylab import title
        t = current_data.t
        timestr = timeformat(t)
        title('%s after earthquake' % timestr)


    #-----------------------------------------
    # Figure for surface
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Domain and transects', figno=0)
    plotfigure.kwargs = {'figsize':(11,7)}
    plotfigure.show = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('pcolor')
    #plotaxes.axescmd = 'axes([.1,.4,.8,.5])'
    plotaxes.axescmd = 'axes([.1,.1,.4,.8])'
    plotaxes.title = 'Surface'
    #plotaxes.xlimits = [-122.4, -122.16]
    #plotaxes.ylimits = [47.4, 47.8]

    x1_tr1 = -122.29
    x2_tr1 = -122.215
    y1_tr1 = 47.57
    y2_tr1 = 47.705
    
    x1_tr2 = -122.21
    x2_tr2 = -122.265
    y1_tr2 = 47.4925
    y2_tr2 = 47.545
        
    def aa_transects(current_data):
        from pylab import ticklabel_format, xticks, plot, text, gca,cos,pi
        title_hours(current_data)
        ticklabel_format(useOffset=False)
        xticks(rotation=20)
        plot([x1_tr1, x2_tr1], [y1_tr1, y2_tr1], 'w')
        plot([x1_tr2, x2_tr2], [y1_tr2, y2_tr2], 'w')
        text(x2_tr1-0.01,y2_tr1+0.005,'Transect 1',color='w',fontsize=8)
        text(x1_tr2-0.01,y1_tr2-0.008,'Transect 2',color='w',fontsize=8)
        gca().set_aspect(1./cos(48*pi/180.))
        #addgauges(current_data)


    plotaxes.afteraxes = aa_transects


    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    #plotitem.plot_var = geoplot.surface
    plotitem.plot_var = surface_or_depth_lake
    plotitem.pcolor_cmap = geoplot.tsunami_colormap
    plotitem.pcolor_cmin = cmin
    plotitem.pcolor_cmax = cmax
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0,0,0,0]
    plotitem.amr_data_show = [1,1,1,1,1,0,0]

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = geoplot.land
    plotitem.pcolor_cmap = geoplot.land_colors
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = cmax_land
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0]
    plotitem.amr_patchedges_show = [0,0,0,0]
    plotitem.amr_data_show = [1,1,1,1,1,0,0]

    # add contour lines of bathy if desired:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    #plotitem.show = False
    plotitem.plot_var = geoplot.topo
    plotitem.contour_levels = [sea_level]
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':0.5}
    plotitem.amr_contour_show = [0,1,0,0]  # only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0


    #-----------------------------------------
    # Plots along transect:
    #-----------------------------------------

    eta1 = lambda q: q[3,:,:]
    B1 = lambda q: q[3,:,:]-q[0,:,:]
    
    def plot_xsec(current_data):
        import matplotlib.pyplot as plt
        import numpy
        import gridtools
        from clawpack.pyclaw import Solution

        framesoln = current_data.framesoln
        
        topo_color = [.8,1,.8]
        water_color = [.5,.5,1]
        
        plt.figure(0)
        
        # Transect 1:
        plt.axes([.55,.5,.4,.3])
        xlon = numpy.linspace(x1_tr1, x2_tr1, 1000)
        yout = numpy.linspace(y1_tr1, y2_tr1, 1000)
        xout = xlon*numpy.ones(yout.shape)    
        eta = gridtools.grid_output_2d(framesoln, eta1, xout, yout)        
        topo = gridtools.grid_output_2d(framesoln, B1, xout, yout)
        
        eta = numpy.where(eta>topo, eta, numpy.nan)


        plt.fill_between(yout, eta, topo, color=water_color)
        plt.fill_between(yout, topo, -10000, color=topo_color)
        plt.plot(yout, eta, 'b')
        plt.plot(yout, topo, 'g')
        plt.plot(yout, sea_level + 0*topo, 'k--')
        #plt.xlim(47.5,47.8)
        plt.ylim(ylim_transects)
        plt.ylabel('meters')
        plt.grid(True)
        timestr = timeformat(framesoln.t)
        plt.title('Elevation on Transect 1')
        
        # Transect 2:
        plt.axes([.55,.1,.4,.3])
        xlon = numpy.linspace(x1_tr2, x2_tr2, 1000)
        yout = numpy.linspace(y1_tr2, y2_tr2, 1000)
        xout = xlon*numpy.ones(yout.shape)    
        eta = gridtools.grid_output_2d(framesoln, eta1, xout, yout)        
        topo = gridtools.grid_output_2d(framesoln, B1, xout, yout)
        
        eta = numpy.where(eta>topo, eta, numpy.nan)
        topo_color = [.8,1,.8]
        water_color = [.5,.5,1]

        plt.fill_between(yout, eta, topo, color=water_color)
        plt.fill_between(yout, topo, -10000, color=topo_color)
        plt.plot(yout, eta, 'b')
        plt.plot(yout, topo, 'g')
        plt.plot(yout, sea_level + 0*topo, 'k--')
        #plt.xlim(47.5,47.8)
        plt.ylim(ylim_transects)
        plt.ylabel('meters')
        plt.grid(True)
        timestr = timeformat(framesoln.t)
        plt.title('Elevation on Transect 2')

    plotdata.afterframe = plot_xsec

    #-----------------------------------------
    # Figure for zoomed area
    #-----------------------------------------

    # To use, set the limits as desired and set `plotfigure.show = True`
    x1,x2,y1,y2 = [-122.23, -122.2, 47.69, 47.71]
    plotfigure = plotdata.new_plotfigure(name="zoomed area", figno=11)
    plotfigure.show = False
    plotfigure.kwargs = {'figsize': (8,7)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.scaled = False

    plotaxes.xlimits = [x1, x2]
    plotaxes.ylimits = [y1, y2]

        
    def aa(current_data):
        from pylab import ticklabel_format, xticks, gca,cos,pi
        title_hours(current_data)
        ticklabel_format(useOffset=False)
        xticks(rotation=20)
        gca().set_aspect(1./cos(48*pi/180.))
        
    plotaxes.afteraxes = aa

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    #plotitem.plot_var = geoplot.surface
    plotitem.plot_var = surface_or_depth_lake
    plotitem.pcolor_cmap = geoplot.tsunami_colormap
    plotitem.pcolor_cmin = cmin
    plotitem.pcolor_cmax = cmax
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.patchedges_show = 0

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = geoplot.land
    plotitem.pcolor_cmap = geoplot.land_colors
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = cmax_land
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0]
    plotitem.patchedges_show = 0


    #-----------------------------------------
    # Figures for gauges
    #-----------------------------------------
    
    time_scale = 1./3600.
    time_label = 'hours'
    
    plotfigure = plotdata.new_plotfigure(name='gauge depth', figno=300, \
                    type='each_gauge')
    #plotfigure.clf_each_gauge = False

    def setglimits_depth(current_data):
        from pylab import xlim,ylim,title,argmax,show,array,ylabel
        gaugeno = current_data.gaugeno
        q = current_data.q
        depth = q[0,:]
        t = current_data.t
        g = current_data.plotdata.getgauge(gaugeno)
        level = g.level
        maxlevel = max(level)

        #find first occurrence of the max of levels used by
        #this gauge and set the limits based on that time
        argmax_level = argmax(level)
        xlim(time_scale*array(t[argmax_level],t[-1]))
        ylabel('meters')
        min_depth = depth[argmax_level:].min()
        max_depth = depth[argmax_level:].max()
        ylim(min_depth-0.5, max_depth+0.5)
        title('Gauge %i : Flow Depth (h)\n' % gaugeno + \
              'max(h) = %7.3f,    max(level) = %i' %(max_depth,maxlevel))    
        #show()

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.time_scale = time_scale
    plotaxes.time_label = time_label

    # Plot depth as blue curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = 'b-'

    ## Set the limits and the title in the function below
    plotaxes.afteraxes = setglimits_depth

    plotfigure = plotdata.new_plotfigure(name='gauge surface eta', figno=301, \
                    type='each_gauge')
    #plotfigure.clf_each_gauge = False

    def setglimits_eta(current_data):
        from pylab import xlim,ylim,title,argmax,show,array,ylabel
        gaugeno = current_data.gaugeno
        q = current_data.q
        eta = q[3,:]
        t = current_data.t
        g = current_data.plotdata.getgauge(gaugeno)
        level = g.level
        maxlevel = max(level)

        #find first occurrence of the max of levels used by
        #this gauge and set the limits based on that time
        argmax_level = argmax(level) #first occurrence of it
        xlim(time_scale*array(t[argmax_level],t[-1]))
        ylabel('meters')
        min_eta = eta[argmax_level:].min()
        max_eta = eta[argmax_level:].max()
        ylim(min_eta-0.5,max_eta+0.5)
        title('Gauge %i : Surface Elevation (eta)\n' % gaugeno + \
              'max(eta) = %7.3f,    max(level) = %i' %(max_eta,maxlevel))
        #show()

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.time_scale = time_scale
    plotaxes.time_label = time_label
    
    # Plot surface (eta) as blue curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 3
    plotitem.plotstyle = 'b-'

    ## Set the limits and the title in the function below
    plotaxes.afteraxes = setglimits_eta

    plotfigure = plotdata.new_plotfigure(name='speed', figno=302, \
                    type='each_gauge')
    #plotfigure.clf_each_gauge = False

    def speed(current_data):
        from numpy import sqrt, maximum, where
        q   = current_data.q
        h   = q[0,:]
        hu  = q[1,:]
        hv  = q[2,:]
        s   = sqrt(hu**2 + hv**2) / maximum(h,0.001)
        s   = where(h > 0.001, s, 0.0)
        return s

    def setglimits_speed(current_data):
        from pylab import xlim,ylim,title,argmax,show,array,ylabel
        gaugeno = current_data.gaugeno
        s = speed(current_data)
        t = current_data.t
        g = current_data.plotdata.getgauge(gaugeno)
        level = g.level
        maxlevel = max(level)

        #find first occurrence of the max of levels used by
        #this gauge and set the limits based on that time
        argmax_level = argmax(level) #first occurrence of it
        xlim(time_scale*array(t[argmax_level],t[-1]))
        ylabel('meters/sec')
        min_speed = s[argmax_level:].min()
        max_speed = s[argmax_level:].max()
        ylim(min_speed-0.5,max_speed+0.5)
        title('Gauge %i : Speed (s)\n' % gaugeno + \
              'max(s) = %7.3f,    max(level) = %i' %(max_speed,maxlevel))
        #show()

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.time_scale = time_scale
    plotaxes.time_label = time_label

    # Plot speed (s) as blue curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = speed
    plotitem.plotstyle = 'b-'

    ## Set the limits and the title in the function below
    plotaxes.afteraxes = setglimits_speed


    #-----------------------------------------
    # Figures for fgmax plots
    #-----------------------------------------
    # Note: need to move fgmax png files into _plots after creating with
    #   python run_process_fgmax.py
    # This just creates the links to these figures...

    if 0:
        ### Putting them in _other_figures with the proper name as a link
        ### Can run process fgmax either before or after setplot now.
        otherfigure = plotdata.new_otherfigure(name='max depth',
                        fname='_other_figures/%s_%s_h_onshore.png' \
                                % (params.loc,params.event))
        otherfigure = plotdata.new_otherfigure(name='max depth on GE image',
                        fname='_other_figures/%s_%s_h_onshore_GE.png' \
                                % (params.loc,params.event))
        otherfigure = plotdata.new_otherfigure(name='max speed',
                        fname='_other_figures/%s_%s_speed.png' \
                                % (params.loc,params.event))

    # Plots of timing (CPU and wall time):

    def make_timing_plots(plotdata):
        import os
        from clawpack.visclaw import plot_timing_stats
        try:
            timing_plotdir = plotdata.plotdir + '/_timing_figures'
            os.system('mkdir -p %s' % timing_plotdir)
            units = {'comptime':'hours', 'simtime':'hours', 'cell':'billions'}
            plot_timing_stats.make_plots(outdir=plotdata.outdir, make_pngs=True,
                                          plotdir=timing_plotdir, units=units)
            os.system('cp %s/timing.* %s' % (plotdata.outdir, timing_plotdir))
        except:
            print('*** Error making timing plots')

    otherfigure = plotdata.new_otherfigure(name='timing',
                    fname='_timing_figures/timing.html')
    otherfigure.makefig = make_timing_plots


    #-----------------------------------------

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'        # list of frames to print
    plotdata.print_gaugenos = 'all'          # list of gauges to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True                 # make multiple frame png's at once

    return plotdata
