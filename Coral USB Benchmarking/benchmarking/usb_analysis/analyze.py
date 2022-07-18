results_folder = "results/"

class Operation:
    def __init__(self, path, op_name):
        self.path = path
        self.op_name  = op_name
        self.samples  = []
        self.mean = 0
        self.median = 0
        self.std_dev = 0

    def get_basic_info(self):
        from statistics import mean, median, stdev

        data = self.samples
        total = len(self.samples)

        self.std_dev = stdev(data)
        self.mean = mean(data)
        self.median = median(data)

    def best_fit_distribution(self, bins=1000, ax=None):
        """Model data by finding best fit distribution to data"""

        import warnings
        import pandas as pd
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

        import pandas as pd
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
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        data = self.samples

        plt.hist(data, bins = bins)
        plt.savefig(self.path + "/" + self.op_name + "_hist.png")

    dist_name = None
    best_dist = None
    best_params = None
    pdf = None


def log_pdfs(op_classes):
    import csv

    for op in op_classes:
        csv_file = f"{op.path}/Analysis.csv"

        with open(csv_file, 'w') as csvfile:
                fw = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

                fw.writerow(["num_of_samples", 
                             "mean", "median", 
                             "std_dev", "pdf"])

                fw.writerow([str(len(op.samples)), 
                             str(op.mean), str(op.median),
                             str(op.std_dev), str(op.dist_name)])


def retreive_stat_info_op(op_classes):
    for op in op_classes:
        op.get_basic_info()
        op.best_fit_distribution()
        op.make_pdf()
        op.plot_all()


def store_op_info(op_classes):
    from utils import parse_csv

    for op in op_classes:
        results_csv = op.path + "/Results.csv"
        op.samples = parse_csv(results_csv)


def init_op_classes(models_folder):
    import os
    from os import listdir
    from os.path import isfile,isdir,join

    op_classes = []

    for f in listdir(models_folder):
        f_path = models_folder + f
        if isdir(f_path):
            op_tmp = Operation(f_path, f)
            op_classes.append(op_tmp)

    return op_classes


def tflite_results_analysis():
    import logging
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

    global results_folder
    log.info("Analyzing cpu results...")
    cpu_op_classes = init_op_classes(results_folder + "cpu/")
    store_op_info(cpu_op_classes)
    retreive_stat_info_op(cpu_op_classes)
    log_pdfs(cpu_op_classes)

    log.info("Analyzing edge results...")
    edge_op_classes = init_op_classes(results_folder + "edge/")
    store_op_info(edge_op_classes)
    retreive_stat_info_op(edge_op_classes)
    log_pdfs(edge_op_classes)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--delegate', required=False, 
                        default="cpu", 
                        help='Delegates: cpu or edge_tpu.')

    parser.add_argument('-m', '--mode', required=False, 
                        default="Group", 
                        help='Plot all hardware analysis.')

    args = parser.parse_args()

    if (args.mode == "Group"):
        cpu_op_classes = init_op_classes(results_folder + "cpu/")
        store_op_info(cpu_op_classes)
        retreive_stat_info_op(cpu_op_classes)
        log_pdfs(cpu_op_classes)

        edge_op_classes = init_op_classes(results_folder + "edge/")
        store_op_info(edge_op_classes)
        retreive_stat_info_op(edge_op_classes)
        log_pdfs(edge_op_classes)
    else:
        gen_op_classes = init_op_classes(f"{results_folder}{args.delegate}/")
        store_op_info(gen_op_classes)
        retreive_stat_info_op(gen_op_classes)

