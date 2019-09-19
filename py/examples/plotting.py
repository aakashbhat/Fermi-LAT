
# update plotting parameters from a dictionary
params = {'axes.labelsize': 25, #unit
          'axes.titlesize': 25,  #title
          'font.size': 20,
          'legend.fontsize': 15,
          'xtick.labelsize':18, #ticks
          'ytick.labelsize':20,
          #'text.usetex': True,
          #'figure.figsize': fig_size,
          'xtick.major.size' : 6,
          'ytick.major.size' : 6,
          'xtick.minor.size' : 3,
          'ytick.minor.size' : 3,
          'figure.subplot.left' : 0.05,
          'figure.subplot.right' : 0.97,
          'figure.subplot.bottom' : 0.15,
          'figure.subplot.top' : 0.9
        }
pyplot.rcParams.update(params)

# or directly update individual entries
pyplot.rcParams['xtick.labelsize'] = 20

