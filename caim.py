import argparse
import pandas as pd
import numpy as np
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings('error')


class CAIM(object):
    def __init__(self):
        self.OriginalData = None
        self.caim_results = None

    def print_interval_results(self):
        if self.caim_results is None:
            print("CAIM object not fitted")
            return

        for col in sorted(self.caim_results.keys()):
            caim_val, intervals = self.caim_results[col]
            print("Column: %s" % str(col))
            print("\tCAIM Value: %f" % caim_val)
            print("\tIntervals: %s" % str(intervals))

    def _create_init_data(self, X, Y):
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
        try:
            Y = Y.astype(int).copy()
        except:
            y_vals = Y.unique()
            self.y_mapping = dict(zip(y_vals, range(0, len(y_vals))))
            Y = Y.apply(self.y_mapping.__getitem__).astype(int).copy()

        self.OriginalData = X.join(Y)

        self.X = X
        self.num_Y = Y
        self.C, self.F, self.M = c, f, m

    @staticmethod
    def discretize_series(series, interval):
        f = lambda x: interval[x] if x != 0 else interval[1]
        binned = pd.Series(np.digitize(series,
                                       interval,
                                       right=True)).apply(f)
        return binned

    def _do_run_feature(self, feature_series, class_series, **kwargs):
        """Wrapper for using exec run_feature in parallel proc"""
        return CAIM._run_feature(feature_series, class_series, **kwargs)

    @staticmethod
    def _run_feature(feature_series, class_series):
        print("Running %s" % str(feature_series.name))
        k = 1

        input_data = pd.DataFrame([feature_series, class_series]).T
        feature_name = feature_series.name
        class_name   = class_series.name

        num_classes = len(class_series.unique())-1
        done = False

        remaining_int = np.array(feature_series.unique()).astype(float)
        remaining_int.sort()

        if len(remaining_int) == 2:
            remaining_int = np.insert(remaining_int, 1, remaining_int.max()/2.0)
        elif len(remaining_int) == 1:
            msg = "Feature %s has only one unique value" % feature_series.name
            raise ValueError(msg)

        # Starting interval is end to end set
        disc_interval = np.array([remaining_int[0], remaining_int[-1]])
        remaining_int = remaining_int[1:-1]
        f = lambda x: CAIM.compute_caim(CAIM.build_quanta(input_data, x, feature_name, class_name))

        global_caim = 0
        while not done:

            ints_and_caims = ((add_int, f(np.sort(np.insert(disc_interval, 0, add_int)))) for add_int in remaining_int)
            max_int_and_caim = max(ints_and_caims, key=lambda x: x[1])

            current_caim = max_int_and_caim[1]
            current_int = np.sort(np.insert(disc_interval, 0, max_int_and_caim[0]))
            current_add_int = max_int_and_caim[0]
            better_caim = current_caim > global_caim

            if better_caim:
                disc_interval = current_int
                global_caim = current_caim
            if k < num_classes or better_caim:
                remaining_int = remaining_int[remaining_int != current_add_int]
                k += 1
            else:
                done = True

            if len(remaining_int) == 0:
                done = True

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

    def predict(self, X):
        return X.apply(lambda x: CAIM.discretize_series(x, self.caim_results[x.name][1]))

    def fit(self, X, Y, n_jobs=-1, verbose=False):
        self._create_init_data(X, Y)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Don't start more processes than will be needed
        if n_jobs > len(X.columns):
            n_jobs = len(X.columns)

        if verbose:
            print("Total features: %s" % self.F)

        f = partial(self._do_run_feature, class_series=self.num_Y)
        cols = sorted(list(X.columns))
        feature_columns = [X[c] for c in cols]

        if verbose:
            print("Running with %d processes" % n_jobs)

        # Run in parallel across multiple processes
        pool = multiprocessing.Pool(n_jobs)
        res = pool.map(f, feature_columns)

        # Assemble results
        self.caim_results = dict(zip(cols, res))

        return self

    def fit_old(self, X, Y):
        """Old Single process version for debugging"""
        self._create_init_data(X, Y)

        # TODO: Parallelize with fork server?
        print("Total features: %s" % self.F)
        results = dict()
        for f_name in X.columns:
            print("Running: %s" % str(f_name))
            results[f_name] = self._run_feature(self.OriginalData[f_name], Y)

        self.caim_results = results

        return self

def parse_field_arguments(all_columns, target_arg_str, verbose=False):
    """Return tuple of feature columns and target column"""
    try:
        target = all_columns[int(target_arg_str)]
    except ValueError:
        target = target_arg_str

    if verbose:
        print("Target Column: %s" % str(target))

    features = list(all_columns).copy()
    if target in features:
        features.remove(target)
    else:
        del features[target]

    return features, target

if __name__ == "__main__":
    desc = "CAIM Algorithm Command Line Tool and Library"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('input_data', metavar='input_file',
                        type=str, nargs=1,
                        help="CSV input data file")

    parser.add_argument('-t', '--target-field',
                        dest='target_field', default=None,
                        help=("Target fields as integers (0-indexed) " +
                              "or strings corresponding to column names." +
                              "Negative indices (e.g. -1) are allowed."))

    parser.add_argument('-o', '--output-path',
                        dest='output_path', default=None,
                        help="File path to write discrete form of data in CSV format")

    parser.add_argument('-H', '--header',
                        dest='header', default=False, action='store_true',
                        help="Use first row as column name rows")

    parser.add_argument('-q', '--quiet',
                        dest='quiet', default=False, action='store_true',
                        help="Minimal information is printed to STDOUT")

    args = parser.parse_args()
    input_data = args.input_data[0]
    header = 0 if args.header else None

    if not args.quiet:
        print("Loading data from %s" % input_data)

    input_df = pd.read_csv(input_data, header=header)

    feature_fields, target_field = parse_field_arguments(input_df.columns,
                                                         args.target_field,
                                                         not args.quiet)
    if not args.quiet:
        print("Feature:\n%s" % str(feature_fields))
        print("Target:\n%s" % str(target_field))

    caim = CAIM().fit(input_df[feature_fields],
                      input_df[target_field],
                      -1, not args.quiet)

    final_data = caim.predict(input_df[feature_fields]).join(input_df[target_field])

    if not args.quiet:
        #print("Intervals:\n%s\n" % str(caim.caim_results))
        caim.print_interval_results()
    if not args.quiet:
        print("New Dataset:\n------------\n%s\n" % str(final_data))

    if args.output_path:
        final_data.to_csv('%s' % args.output_path,
                          index=None, header=True if header is not None else False)

