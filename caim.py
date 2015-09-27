import argparse
import pandas as pd
import numpy as np
import math
import warnings
import multiprocessing
from bisect import bisect_left, bisect, bisect_right
from functools import partial
from joblib import Parallel, delayed
from pprint import pprint
warnings.filterwarnings('error')


class CAIM(object):
    def __init__(self):
        pass

    def _create_init_data(self, X, Y):
        full_columns = list(X.columns) + [Y.name]
        if isinstance(Y, pd.Series):
            c = 1
        else:
            c = len(Y.columns)

        if isinstance(X, pd.Series):
            f = 1
        else:
            f = len(X.columns)

        assert len(X) == len(Y)
        m = len(X)
        Y = Y.astype(int)
        self.OriginalData = X.join(Y)

        discrete_data = pd.DataFrame(0,
                                     index=np.arange(m),
                                     columns=full_columns)
        #discrete_data[Y.columns] = Y
        #discrete_data[Y.columns] = Y
        #max_num_f = math.floor(m/(3*c))

        self.C, self.F, self.M = c, f, m
        #self.DiscreteData      = discrete_data
        #self.DiscretizationSet_dict = dict()
        #self.DiscretizationSet = None
        #self.MaxNumF           = max_num_f

    @staticmethod
    def discretize_series(series, interval):
        f = lambda x: interval[x] if x != 0 else interval[1]
        binned = pd.Series(np.digitize(series,
                                       interval,
                                       right=True)).apply(f)
        return binned

    def _do_run_feature(self, feature_series, class_series, **kwargs):
        return CAIM._run_feature(feature_series, class_series, **kwargs)

    @staticmethod
    def _run_feature(feature_series, class_series):
        print("Running %s" % str(feature_series.name))
        k = 1

        input_data = pd.DataFrame([feature_series, class_series]).T
        feature_name = feature_series.name
        class_name   = class_series.name

        num_classes = len(class_series.unique())-1
        interval_idx = 1
        done = False

        remaining_int = np.array(feature_series.unique()).astype(float)
        remaining_int.sort()

        if len(remaining_int) == 2:
            remaining_int = np.insert(remaining_int, 1, remaining_int.max()/2.0)
        elif len(remaining_int) == 1:
            msg="Feature %s has only one unique value" % feature_series.name
            raise ValueError(msg)

        # Starting interval is end to end set
        disc_interval = np.array([remaining_int[0], remaining_int[-1]])
        remaining_int = remaining_int[1:-1]
        f = lambda x: CAIM.build_quanta(input_data, x, feature_name, class_name)

        global_caim = 0
        while not done:
            current_caim = -np.inf
            current_int = np.nan

            possibble_int = pd.Series([np.sort(np.insert(disc_interval, 0, add_int)) for add_int in remaining_int])
            caims = possibble_int.apply(f).apply(CAIM.compute_caim)
            caim_maxidx = caims.idxmax()

            #caims_ints = pd.concat([possibble_int, caims], axis=1)
            #best = caims_ints.iloc[caims_ints[1].idxmax()]

            #current_caim = best[1]
            #current_int = best[0]
            current_caim = caims[caim_maxidx]
            current_int = possibble_int[caim_maxidx]
            current_add_int = remaining_int[caim_maxidx]
            better_caim = current_caim > global_caim
            if better_caim:
                #print("Current CAIM: %f" % current_caim)
                #print("Global CAIM: %f" % global_caim)
                disc_interval = current_int
                global_caim = current_caim
            if k < num_classes or better_caim:
                #print(current_add_int)
                remaining_int = remaining_int[remaining_int != current_add_int]

                k += 1
            else:
                done = True

            if len(remaining_int) == 0:
                done = True
        #print("Best CAIM: %f" %global_caim)
        #print("Interval : %s" % str(disc_interval))
        return global_caim, disc_interval

    @staticmethod
    def build_quanta(input_data, intervals, feature_column, class_column):

        binned = pd.Series(np.digitize(input_data[feature_column],
                                       intervals,
                                       right=True)).apply(lambda x: x if x != 0 else 1)

        binned.name = 'bins'
        grpby = [binned, class_column]
        quanta = input_data.groupby(grpby).count()
        return quanta

    @staticmethod
    def compute_caim(quanta):
        # Get the M_r value (number of values in the bin)
        m_r = quanta.sum(axis=0, level=0)

        # Get Max_r (maximum class count for the bin)
        max_r = quanta.max(axis=0, level=0)

        # This will only count up the number of bins
        # that have some items
        n = len(quanta.index.levels[0])

        return ((max_r**2/m_r).sum()/n).values[0]

    def fit_parallel(self, X, Y):
        self._create_init_data(X, Y)

        # TODO: Parallelize with fork server?
        print("Total features: %s" % self.F)
        f = partial (self._do_run_feature, class_series=Y)
        cols = sorted(list(X.columns))
        feature_columns = [X[c] for c in cols]

        pool = multiprocessing.Pool(4)
        res = pool.map(f, feature_columns)
        self.caim_results = dict(zip(cols, res))
        #print(cols, res)
        #pprint(self.caim_results)


        #for p in range(0, self.F):
        #    print("Running: %s" % str(p))
        #    self._run_feature(X, Y, p)

        #self.DiscretizationSet = pd.DataFrame(self.DiscretizationSet_dict)
        return self

    def predict(self, X):
        return X.apply(lambda x: CAIM.discretize_series(x, self.caim_results[x.name][1]))

    def fit(self, X, Y):
        self._create_init_data(X, Y)

        # TODO: Parallelize with fork server?
        print("Total features: %s" % self.F)
        results = dict()
        for f_name in X.columns:
            print("Running: %s" % str(f_name))
            results[f_name] = self._run_feature(self.OriginalData[f_name], Y)

        return self


def parse_field_arguments(all_columns, target_arg_str):
    feature_fields = None
    if not '-' in  target_arg_str:
        targets = target_arg_str.split(',')
        try:
            targets_ints = [int(s) for s in target_str]
            targets      = [all_columns[i] for i in targets_ints]
        except:
            pass

    else:
        target_points = target_arg_str.split('-')
        start, end = int(target_points[0]), int(target_points[1])
        targets      = [all_columns[i] for i in range(start, end+1)]

    features = list(set(all_columns) - set(targets))

    return features, targets

if __name__ == "__main__":
    desc = "CAIM Algorithm Command Line Tool and Library"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('input_data', metavar='input_file',
                        type=str, nargs=1,
                        help="CSV input data file")

    parser.add_argument('-t', '--target-fields',
                        dest='target_field', default=None,
                        help=("Target fields as integers (0-indexed) " +
                              "or strings corresponding to column names"))

    parser.add_argument('-o', '--output-base-name',
                        dest='output_base', default=None,
                        help="Base name for outputs: <base>_data.csv and <base>_sets.csv")

    parser.add_argument('-H', '--header',
                        dest='header', default=False, action='store_true',
                        help="Use first row as column")

    parser.add_argument('-v', '--verbose',
                        dest='verbose', default=False, action='store_true',
                        help="Output additional information")

    args = parser.parse_args()

    if args.header:
        input_df = pd.read_csv(args.input_data[0])
    else:
        input_df = pd.read_csv(args.input_data[0], header=None)

    # This mangles the ordering, can make it hard to review output
    feature_fields, target_fields = parse_field_arguments(input_df.columns,
                                                          args.target_field)
    if args.verbose:
        print("Feature:\n%s" % str(feature_fields))
        print("Target:\n%s" % str(target_fields))

    caim = CAIM().fit_parallel(input_df[feature_fields],
                                input_df[target_fields[0]])
    final_data = caim.predict(input_df[feature_fields])
    if args.verbose:
        print("New Dataset:\n------------\n%s\n" % str(final_data))
        #print("Sets:\n-----\n%s" % str(caim.DiscretizationSet))

    if args.output_base:
        final_data.to_csv('%s_data.csv' % args.output_base,
                                 index=None)

#        caim.DiscretizationSet.to_csv('%s_sets.csv' % args.output_base,
#                                       index=None)

