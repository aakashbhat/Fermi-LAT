# setup figure parameters:
# import plotting as my_plot
# my_plot.setup_figure_pars()

from matplotlib import pyplot

def setup_figure_pars():
    # update plotting parameters from a dictionary
    fig_width = 8  # width in inches
    fig_height = 5    # height in inches
    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 16, #unit
              'axes.titlesize': 18,  #title
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize':12, #ticks
              'ytick.labelsize':12,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.12,
              'figure.subplot.right' : 0.98,
              'figure.subplot.bottom' : 0.15,
              'figure.subplot.top' : 0.9,
              'lines.markersize' : 5.5,
              "legend.labelspacing" : 0.3
            }
    pyplot.rcParams.update(params)
    return 0

# One can also directly update individual entries
#pyplot.rcParams['xtick.labelsize'] = 20

