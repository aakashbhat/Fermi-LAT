# setup figure parameters:
# import plotting as my_plot
# my_plot.setup_figure_pars()

import numpy as np
from matplotlib import pyplot
from matplotlib import transforms


def setup_figure_pars():
    # update plotting parameters from a dictionary
    fig_width = 8  # width in inches
    fig_height = 6    # height in inches
    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 18, #unit
              'axes.titlesize': 22,  #title
              'font.size': 14,
              'legend.fontsize': 14,
              'xtick.labelsize':14, #ticks
              'ytick.labelsize':14,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.13,
              'figure.subplot.right' : 0.95,
              'figure.subplot.bottom' : 0.15,
              'figure.subplot.top' : 0.9
            }
    pyplot.rcParams.update(params)
    return 0

# One can also directly update individual entries
#pyplot.rcParams['xtick.labelsize'] = 20

def add_mollview_colorbar(pix_map, label='', vmin=None, vmax=None, 
                              nticks=5, ticks=None, tick_labels=None):
    # bounding box parameters for the main map on healpy mollview
    x0 = 0.02
    y0 = 0.15 # y0=0.185 - original mollview setup
    x1 = 0.98
    y1 = 0.92 # y1=0.95 - original mollview setup
    
    points = [[x0, y0], [x1, y1]]
    bbox_mollview = transforms.Bbox(points)
    
    # determine the values of the ticks and the tick labels
    if ticks is None:
        if vmin is None:
            vmin = np.min(pix_map)
        if vmax is None:
            vmax = np.max(pix_map)
        ticks = np.linspace(vmin, vmax, nticks)
    else:
        nticks = len(ticks)
    if tick_labels is None:
        tick_labels = ['%.2g' % x for x in ticks]
    
    # set the posistion of the map
    plot_ax = pyplot.gcf().get_children()[1]
    plot_ax.set_position(bbox_mollview)
    
    # add the color bar with the custom tick values and labels
    image = plot_ax.get_images()[0]
    cbar = pyplot.colorbar(image, orientation='horizontal',
                           pad=.03, fraction=0.06, aspect=30, shrink=.8,
                           ticks=ticks, label=label)
        
    cbar.set_ticklabels(tick_labels)
    return 0

def mollview_grid_lines():
    ax = pyplot.gcf().get_children()[1]
    axis = ax.axis["lat=0"]
    axis.line.set_linestyle(':')
    axis.line.set_linewidth(.5)

    # get rid of the numbers on the grid lines

    for key in list(ax.axis.keys()):
        ax.axis[key].major_ticklabels.set_visible(False)
        ax.axis[key].set_label('')

    ax.grid()

