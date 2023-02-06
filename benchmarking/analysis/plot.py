import matplotlib.pyplot as plt

def plot(self, bins=1000):
    import os
    import matplotlib

    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)

    data = self._get_data()

    plt.hist(data, bins = bins)
    plt.savefig(os.path.join("results/", f"{self.model_name}_hist.png"))
