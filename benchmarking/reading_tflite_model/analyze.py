import matplotlib
import matplotlib.pyplot as plt
import statsmodels as sm
import pandas as pd

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

edge_op_classes = []
cpu_op_classes = []
gen_op_classes = []

delegate = ""
results_folder = "results/"
delegate_folder = results_folder + delegate

class Operation:
    def __init__(self, path, op_name):
        self.path = path
        self.op_name  = op_name
        self.samples  = []
        self.mean = 0
        self.median = 0
        self.std_dev = 0

    def get_basic_info(self):
        import numpy as np
        import pandas as pd

        data = self.samples
        total = len(self.samples)

        self.std_dev = np.std(data, dtype = np.float64)
        self.mean = np.mean(data, dtype = np.float64)
        self.median = np.median(data)

    def best_fit_distribution(self, bins=1000, ax=None):
        """Model data by finding best fit distribution to data"""

        import warnings
        import numpy as np
        import scipy.stats as st


        data = self.samples

        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Distributions to check
        DISTRIBUTIONS = [        
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk
            #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]

        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        for dist in DISTRIBUTIONS:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    params = dist.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = dist
                        best_params = params
                        best_sse = sse


            except Exception:
                pass
        
        self.dist_name = best_distribution.name
        self.best_dist = best_distribution
        self.best_params = best_params

    def make_pdf(self, size=1000):
        """Generate distributions's Probability Distribution Function """

        import numpy as np
        import scipy.stats as st

        dist = self.best_dist
        params = self.best_params

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
        
        self.pdf = pdf

    def plot_all(self, bins=1000):
        import pandas as pd
        import numpy as np

        data = self.samples

        plt.hist(data, bins = bins)
        plt.savefig(self.path + "/" + self.op_name + "_hist.png")

    dist_name = None
    best_dist = None
    best_params = None
    pdf = None


def log_pdfs(OP_CLASSES):
    for op in OP_CLASSES:
        file = op.path + "/" + op.op_name + "_info.txt"
        with open(file, 'w') as f:
            f.write("Operation: " + op.op_name +  "\n")
            f.write("Number of Samples: " + str(len(op.samples)) +  "\n")
            f.write("Mean: " + str(op.mean) +  "\n")
            f.write("Median: " + str(op.median) +  "\n")
            f.write("Standard Deviation: " + str(op.std_dev) +  "\n")
            f.write("Distribution Name: " + str(op.dist_name) +  "\n")
            f.write("PDF: " + str(op.pdf) +  "\n")

def retreive_stat_info_op(OP_CLASSES):
    for op in OP_CLASSES:
        op.get_basic_info()
        op.best_fit_distribution()
        op.make_pdf()
        op.plot_all()

        pass


def parse_csv(file):
    import csv

    samples = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            samples.append(float(row[1]))

    return samples


def store_op_info(OP_CLASSES):
    for op in OP_CLASSES:
        results_csv = op.path + "/Results.csv"
        op.samples = parse_csv(results_csv)


def init_op_classes(models_folder, OP_CLASSES):
    import os
    from os import listdir
    from os.path import isfile,isdir,join

    for f in listdir(models_folder):
        f_path = models_folder + f
        if isdir(f_path):
            op_tmp = Operation(f_path, f)
            OP_CLASSES.append(op_tmp)


def full_tflite_analysis():
    global cpu_op_classes
    global edge_op_classes

    init_op_classes(args.folder + "cpu/", cpu_op_classes)
    store_op_info(cpu_op_classes)
    retreive_stat_info_op(cpu_op_classes)
    log_pdfs(cpu_op_classes)

    init_op_classes(args.folder + "edge/", edge_op_classes)
    store_op_info(edge_op_classes)
    retreive_stat_info_op(edge_op_classes)
    log_pdfs(edge_op_classes)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--delegate', required=False, 
                        default="cpu", 
                        help='Delegates: cpu or edge_tpu.')

    parser.add_argument('-f', '--folder', required=False, 
                        default="results/", 
                        help='Folder containing results.')

    parser.add_argument('-a', '--all', required=False, 
                        default=True, 
                        help='Plot all hardware analysis.')

    args = parser.parse_args()

    delegate = args.delegate + "/"
    results_folder = args.folder
    results_folder += delegate

    if (not args.all):
        init_op_classes(results_folder, gen_op_classes)
        store_op_info(gen_op_classes)
        retreive_stat_info_op(gen_op_classes)
    else:
        init_op_classes(args.folder + "cpu/", cpu_op_classes)
        store_op_info(cpu_op_classes)
        retreive_stat_info_op(cpu_op_classes)
        log_pdfs(cpu_op_classes)

        init_op_classes(args.folder + "edge/", edge_op_classes)
        store_op_info(edge_op_classes)
        retreive_stat_info_op(edge_op_classes)
        log_pdfs(edge_op_classes)

