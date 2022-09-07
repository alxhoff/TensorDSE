import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from utils.model import Model

class Analyzer:
    def __init__(self, m:Model):
        self.model_path     = m.model_path
        self.model_name     = m.model_name
        self.results        = m.results
        self.timers         = m.timers
        self.mean           = 0.0
        self.median         = 0.0
        self.std_deviation  = 0.0
        self.avg_absolute_deviation  = 0.0

    def _get_data(self):
        data = []
        # results are organized as [iteration, inference time of current iteration]
        for r in self.results:
            data.append(r[1])

        return data

    def get_basic_statistics(self):
        from statistics import mean, median, stdev
        data = self._get_data()
        self.mean           = mean(data)
        self.median         = median(data)
        self.std_deviation  = stdev(data)
        self.avg_absolute_deviation  = (
        ((mean([abs(n - self.mean) for n in data])) / self.mean) *100
        )

        if not self.timers:
            self.usb_statistics = {
                    "valid" : len(self.timers)
            }
            for k in self.timers[0]:
                self.usb_statistics[k]["mean"] = mean(
                        [t[k] for t in self.timers]
                )
                self.usb_statistics[k]["median"] = median(
                        [t[k] for t in self.timers]
                )
                self.usb_statistics[k]["std_deviation"] = stdev(
                        [t[k] for t in self.timers]
                )

    def get_distribution(self, bins=1000, ax=None):
        """Model data by finding best fit distribution to data"""
        import warnings

        data = self._get_data()

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

        best_distribution   = st.norm
        best_parameters     = (0.0, 1.0)
        best_sse            = np.inf

        for dist in DISTRIBUTIONS:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    parameters = dist.fit(data)

                    # Separate parts of parameters
                    arg = parameters[:-2]
                    loc = parameters[-2]
                    scale = parameters[-1]

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
                        best_parameters = parameters
                        best_sse = sse


            except Exception:
                pass

        self.distribution_name  = best_distribution.name
        self.best_distribution  = best_distribution
        self.best_parameters    = best_parameters

    def make_pdf(self, size=1000):
        """Generate distributions's Probability Distribution Function """
        distribution = self.best_distribution
        parameters   = self.best_parameters

        # Separate parts of parameters
        arg = parameters[:-2]
        loc = parameters[-2]
        scale = parameters[-1]

        # Get sane start and end points of distribution
        start   = distribution.ppf(0.01, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.01, loc=loc, scale=scale)
        end     = distribution.ppf(0.99, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = distribution.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        self.pdf = pdf

    def plot(self, bins=1000):
        import os
        import matplotlib

        matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)

        data = self._get_data()

        plt.hist(data, bins = bins)
        plt.savefig(os.path.join("results/", f"{self.model_name}_hist.png"))
