# setup figure parameters:
# import plotting as my_plot
# my_plot.setup_figure_pars()

from matplotlib import pyplot

def setup_figure_pars():
    # update plotting parameters from a dictionary
    fig_width = 8  # width in inches
    fig_height = 5    # height in inches
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
              'figure.subplot.left' : 0.1,
              'figure.subplot.right' : 0.95,
              'figure.subplot.bottom' : 0.1,
              'figure.subplot.top' : 0.9
            }
    pyplot.rcParams.update(params)
    return 0

# One can also directly update individual entries
#pyplot.rcParams['xtick.labelsize'] = 20

