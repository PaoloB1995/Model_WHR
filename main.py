from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from scipy.stats import linregress
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.externals import joblib

import numpy as np

from sklearn.preprocessing import StandardScaler


class DegradationModel:
    def __init__(self, w, w_step, feature_cols):

        self.w = w
        self.w_step = w_step
        self.feature_cols = feature_cols

        self.machine_life_cycles = None
        self.slope = None
        self.intercept = None
        self.r_value = None
        self.p_value = None
        self.std_err = None
        return

    def dist_function(self, u, v):
        return wasserstein_distance(u, v)

    def logistic(self, v):
        return 1 / (1 + np.exp(v))

    def fit(self, base_data, deg_data, verbose=False):
        # self.base_data = base_data.copy()
        self.machine_life_cycles = len(deg_data)

        self.deg_data = deg_data.copy()

        # STANDARDIZATION

        deg_data[deg_data.columns] = StandardScaler().fit_transform(deg_data[deg_data.columns])

        base_std = deg_data.iloc[:self.w]

        self.base_data = base_std.copy()

        x = []
        degs = []

        for idx in range(0, deg_data.__len__() - self.w, self.w_step):
            d = 0
            for f in self.feature_cols:
                b_data = base_std[f].copy()
                # b_data = (b_data - b_data.mean())/b_data.std()

                d_data = deg_data[f].iloc[idx:idx + self.w].copy()
                # d_data = (d_data - d_data.mean())/d_data.std()

                d += self.dist_function(b_data, d_data)

            print(idx, idx + self.w, "->", d) if verbose else None

            x.append(idx + self.w)
            degs.append(d / len(self.feature_cols))

        if verbose:
            fig1, ax0 = plt.subplots(figsize=(12, 4))
            fig2, ax1 = plt.subplots(figsize=(12, 4))
            fig3, ax2 = plt.subplots(figsize=(12, 4))
            deg_data["kurtosis_4"].plot(label="kurtosis_4", ax=ax0, alpha=0.4, legend=True)
            deg_data["skewness_11"].plot(label="skewness_11", ax=ax0, alpha=0.4, legend=True)
            # deg_data["mean_4"].plot(label="mean_4", ax=ax0, alpha=0.4, legend=True)
            # deg_data["mean_6"].plot(label="mean_6", ax=ax0, alpha=0.4, legend=True)

            for vx in x:
                ax0.axvline(vx, color="gray", linewidth=0.3)
                ax1.axvline(vx, color="gray", linewidth=0.3)
                ax2.axvline(vx, color="gray", linewidth=0.3)
            ax0.set_ylabel("Norm. Kilograms")
            ax0.set_ylim(-10, 10)

            ax1.plot(x, degs, marker=".")
            ax1.set_xlim(0, len(deg_data))
            ax1.set_xlabel("# sample")
            ax1.set_ylabel("Mean Wasserstein Dist.")

            sns.regplot(x, degs, ax=ax2)
            ax2.set_xlim(0, len(deg_data))
            # ax2.set_xticks(x_mesi)
            # ax2.set_xticklabels(['jan', 'feb', 'mar', 'ap', 'ma'], fontsize=12)
            ax2.set_xlabel("# sample")
            ax2.set_ylabel("Degradation trend")
            # ax2.axvline(11500, color="red", linewidth=2.5, linestyle='--')
            # ax2.axvline(25500, color="red", linewidth=2.5, linestyle='--')
            # plt.tight_layout()
            # fig2.savefig("Distances")
            # fig3.savefig("Model")
            plt.show()

        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = linregress(x, degs)
        return self

    def predict(self, data):
        d = 0
        # data[data.columns] = StandardScaler().fit_transform(data[data.columns])
        data_df = (data - self.deg_data.mean()) / self.deg_data.std()

        for f in self.feature_cols:
            b_data = self.base_data[f]
            # b_data = (b_data - b_data.mean())/b_data.std()

            d_data = data_df[f]
            # d_data = (d_data - d_data.mean())/d_data.std()

            d += self.dist_function(b_data, d_data)

        d /= len(self.feature_cols)
        # d = self.dist_function(self.base_data, data)
        deg_cycle = (d - self.intercept) / self.slope
        if deg_cycle < 0:
            deg_cycle = 0
        if deg_cycle > self.machine_life_cycles:
            deg_cycle = self.machine_life_cycles

        degradation = deg_cycle / self.machine_life_cycles * 100
        degradation_cycle = int(self.machine_life_cycles - deg_cycle)
        print(f"{d:.2f} Total Degradation")
        print(f"{degradation:.2f} % Machine Degradation")
        print(f"{int(degradation_cycle)} # Remaining Cycles")

        return degradation, degradation_cycle, d


    def dump(self, path):
        if not path.endswith(".pkl"):
            path += ".pkl"

        return joblib.dump(self, path)

    @staticmethod
    def load(path):
        m = joblib.load(path)
        if m.__class__.__name__ != DegradationModel.__name__:
            raise Exception(
                "Model {} not instance of {}".format(m.__class__.__name__, DegradationModel.__name__))
        return m


def rul_features_to_spark(feature_ids, out_path):
    from pyspark.sql import SparkSession
    import pandas as pd
    spark = SparkSession.builder \
        .master("local") \
        .appName("temp_app").getOrCreate()

    features_df = pd.DataFrame(feature_ids)

    features_s_df = spark.createDataFrame(features_df)
    features_s_df.show()

    features_s_df.write.option("header", True).csv(out_path)


def model_to_spark(model, out_path):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .master("local") \
        .appName("temp_app").getOrCreate()

    print(type(model))
    m_rdd = spark.sparkContext.parallelize([model])
    m_rdd.foreach(lambda x: print(x))
    m_rdd.saveAsPickleFile(out_path)


def load_model_from_spark(model_path):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .master("local") \
        .appName("temp_app").getOrCreate()

    m_rdd = spark.sparkContext.pickleFile(model_path)
    return m_rdd.take(1)[0]



def run_whirlpool_deg_model():

    data = pd.read_csv("C:\\Users\\39338\\Downloads\\Force_Close.csv")
    sel = pd.read_csv("C:\\Users\\39338\\Downloads\\Force_Close_selected.csv")

    data=data.dropna()
    data = data.sort_values('TS')
    data = data.reset_index().drop(columns='index')

    feature = list(sel['feature'])
    feature.append('TS')
    data = data[feature]

    data_test = data[data['TS'].str.startswith('2019-09') | data['TS'].str.startswith('2019-10') | data['TS'].str.startswith('2019-11')]
    data = data[~data['TS'].str.startswith('2019-09')]
    data = data[~data['TS'].str.startswith('2019-10')]
    data = data[~data['TS'].str.startswith('2019-11')]
    data = data.reset_index().drop(columns='index')

    data_1 = data[data.index.isin(range(30491))]
    data_2 = data[~data.index.isin(range(30491))]

    data_1.pop('TS')
    data_2.pop('TS')
    data_test.pop('TS')

    data_tot = data_1.copy()

    w = 2500
    w_step = 1250

    base = data_tot.iloc[:w]
    d_model = DegradationModel(w, w_step, data_tot.columns)
    d_model.fit(base, data_tot.reset_index().drop(columns='index'), verbose=True)


    # SALVATAGGIO DEL MODELLO
    # d_model.dump("/Users/francescoventura/PycharmProjects/SERENA_analytics/misc/models/20201201_whirlpool_rul_model/rul_no_norm_model.pkl")
    #
    # model_to_spark(
    #     d_model, #LinearDegradationModel.load("/Users/francescoventura/PycharmProjects/SERENA_analytics/misc/models/20201116_whirlpool_rul_model/rul_no_norm_model.pkl"),
    #     "/Users/francescoventura/PycharmProjects/SERENA_analytics/misc/models/20201201_whirlpool_rul_model/20201201000000_whirlpool-mimosa_rul_model")
    # rul_features_to_spark({"idx": list(range(len(data_tot.columns))), "feature": feature}, "/Users/francescoventura/PycharmProjects/SERENA_analytics/misc/models/20201201_whirlpool_rul_model/20201201000000_whirlpool-mimosa_rul_features")


    # UTILIZZARE TEST_TOT SE SI VUOLE TESTARE ANCHE SU DATI SET-OTT-NOV 2019
    frames_test = [data_tot, data_test]
    test_tot = pd.concat(frames_test)
    test_tot = test_tot.reset_index().drop(columns='index')

    deg_trend_true = []
    for idx in range(0, len(data_tot)-2500, 1250):
        print("Real value")
        d_true, d_cycle, d = d_model.predict(data_tot[data_tot.columns].iloc[idx:idx+2500])
        deg_trend_true.append(d)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(range(len(deg_trend_true)), deg_trend_true)
    plt.show()


if __name__ == '__main__':
    run_whirlpool_deg_model()