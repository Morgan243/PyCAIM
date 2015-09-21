import argparse
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('error')
#import pdb

class CAIM(object):
    def __init__(self):
        pass

    def _create_init_data(self, X, Y):
        full_columns = list(X.columns) + list(Y.columns)
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
        self.OriginalData = X.join(Y)

        discrete_data = pd.DataFrame(0,
                                    index=np.arange(m),
                                    columns=full_columns)
        discrete_data[Y.columns] = Y
        max_num_f = math.floor(m/(3*c))
        #DiscretizationSet = pd.DataFrame(0,
        #                                 index = np.arange(MaxNumF),
        #                                 columns = list(X.columns))
        #DiscretizationSet = np.array(
        #    [[]]*len(X.columns)
        #)

        self.C, self.F, self.M = c, f, m
        self.DiscreteData      = discrete_data
        self.DiscretizationSet_dict = dict()
        self.DiscretizationSet = None
        self.MaxNumF           = max_num_f

    def _run_feature(self, X, Y, p):
        SortedInterval = (pd.Series(X[X.columns[p]].unique())
                          .sort(inplace=False))

        Len = len(SortedInterval)-1
        B = pd.Series(0.0,
                      index=np.arange(Len))

        for q in range(0, Len):
            B[q] = (SortedInterval[q] + SortedInterval[q+1])/2.0

        D = pd.Series(0.0,
                      index=np.arange(self.MaxNumF))
        GlobalCAIM = -np.inf
        k = -1
        done = False
        while not done:
            CAIM  = -np.inf
            Local = 0
            for q in range(0, Len):
                if not (D[:k+1] == B[q]).any():
                    DTemp = D.copy(deep=True)
                    DTemp[k+1] = B[q]
                    DTemp[:k+2] = DTemp[:k+2].sort(inplace=False,
                                                   ascending=True)

                    CAIMValue = self.CAIM_eval(self.OriginalData,
                                               Y.columns, p, DTemp[:k+2])

                    if CAIM < CAIMValue:
                        CAIM = CAIMValue
                        Local = q

            if GlobalCAIM < CAIM and (k + 1) < self.MaxNumF:
                GlobalCAIM = CAIM
                k += 1
                D[k] = B[Local]
                D[:k+1] = D[:k+1].sort(inplace=False,
                                   ascending=True).copy()
            elif (k + 1) <= self.MaxNumF and (k + 1)<= self.C:
                k += 1
                D[k] = B[Local]
                D[:k+1] = D[:k+1].sort(inplace=False,
                                   ascending=True).copy()
            else:
                done = True

        self.DiscretizationSet_dict[p] = D[:k+1]
        self.DiscreteData.ix[:,p],_ = self.discrete_with_interval(self.OriginalData,
                                                                  Y.columns, p, D[:k+1])

    @staticmethod
    def intermediate_caim_compute(x):
        return 0 if x[0] < 0 else (x[0]**2)/x.sums


    @staticmethod
    def CAIM_eval(original_data, class_names, feature_name, discrete_interval):
        k = len(discrete_interval)
        discrete_data, quanta_matrix = CAIM.discrete_with_interval(original_data,
                                                                   class_names, feature_name,
                                                                   discrete_interval)
        #quanta_max = quanta_matrix.max().reset_index()
        #quanta_max['sums'] = quanta_matrix.sum()
        #caim_value = quanta_max.iloc[:k].apply(CAIM.intermediate_caim_compute, axis=1).sum()

        quanta_max_sum = pd.DataFrame({0:quanta_matrix.max(),
                                       'sums':quanta_matrix.sum()})
        caim_value = quanta_max_sum.iloc[:k].apply(CAIM.intermediate_caim_compute, axis=1).sum()

        return caim_value/k

    @staticmethod
    def within_interval(row, interval):
        match = interval[interval >= row]
        if len(match) > 0:
            return match.index[0]
        else:
            return len(interval)

    @staticmethod
    def discrete_with_interval(original_data, class_fields,
                                 column, discrete_interval):

        #if not isinstance(discrete_interval, pd.Series):
        if type(discrete_interval) != pd.Series:
            discrete_interval = pd.Series([discrete_interval])

        k = len(discrete_interval)

        num_classes = len(class_fields)
        column_name = original_data.columns[column]

        # TODO: Use discrete_interval as an np.array rather than series requires changes to within_interval
        discrete_data = original_data[column_name].apply(lambda x: CAIM.within_interval(x, discrete_interval))

        CState = num_classes
        FState = k + 1

        is_one = original_data[class_fields] == 1
        class_vals = is_one.reset_index()

        # Compute the quanta_matrix columns per each input row
        class_vals['cols'] = class_vals['index'].apply(lambda x: int(discrete_data[x]))
        # Compute the quanta_matrix row per each input row
        class_vals['rows'] = (original_data[class_fields] * np.arange(num_classes)).T.sum()

        # Build quanta_matrix as np.array
        # Matlab implementation lets the quanta_matrix auto-resize as values are added in below loop
        quanta_matrix = np.array([[0.0]*FState] * (int(class_vals.rows.max()) + 1))

        # TODO: Try pulling and iterating over values for speed gain
        # Build the quanta_matric
        for idx, row in class_vals.iterrows():
            quanta_matrix[int(row.rows), int(row.cols)] += 1

        return discrete_data, pd.DataFrame(quanta_matrix)

    def fit(self, X, Y):
        self._create_init_data(X, Y)

        # TODO: Parallize with fork server?
        print("Num Features: %d" % self.F )
        for p in range(0, self.F):
            print("Working on %s" % str(p))
            self._run_feature(X, Y, p)

        self.DiscretizationSet = pd.DataFrame(self.DiscretizationSet_dict)
        #print(self.DiscreteData)
        #print(self.DiscretizationSet)
        return self


def parse_field_arguments(all_columns, target_arg_str):
    feature_fields = None
    if not '-' in  target_arg_str:
        target_str = target_arg_str.split(',')
        try:
            targets_ints = [int(s) for s in target_str]
            targets      = [all_columns[i] for i in targets_ints]
        except:
            pass
            #targets = [int(s) for s in target_str]

    else:
        target_points = target_arg_str.split('-')
        start, end = int(target_points[0]), int(target_points[1])
        targets      = [all_columns[i] for i in range(start, end+1)]

    features = list(set(all_columns) - set(targets))

    return features, targets

if __name__ == "__main__":
    desc = "CAIM Algorithm Command Line Tool"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('input_data', metavar='input_file',
                        type=str, nargs=1,
                        help="CSV input data file")

    parser.add_argument('-t', '--target-fields',
                        dest='target_field', default=None,
                        help=("Target fields as integers (0-indexed) " +
                              "or strings corresponding to column names"))

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
        input_df = pd.read_csv(args.input_data[0], header = None)

    feature_fields, target_fields = parse_field_arguments(input_df.columns,
                                                          args.target_field)
    if args.verbose:
        print("Feature:\n%s" % str(feature_fields))
        print("Target:\n%s" % str(target_fields))

    caim = CAIM().fit(input_df[feature_fields],
                      input_df[target_fields])

    print("New Dataset:\n------------\n%s\n" % str(caim.DiscreteData))
    print("Sets:\n-----\n%s" % str(caim.DiscretizationSet))

