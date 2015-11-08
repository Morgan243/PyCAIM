CAIM is a supervised discretization method [1] and Python-CAIM is a Python implementation of CAIM. **This is a work in progres, results should be closely inspected**. The goal is to provide both a CLI to discretize data for later use as well as a class for programmatic usage. Pull requests welcome.

There is a [MATLAB implementation](http://www.mathworks.com/matlabcentral/fileexchange/24344-caim-discretization-algorithm) by Guangdi Li and a [Java implementation](http://www.cioslab.vcu.edu/index.html) (Research->Data Mining Toole) by the author.

Current Python-CAIM is working on UCI's Musk1 dataset as well as other toy datasets. Results are validated against the Java implementation (see above).

On performance, the Java implementation has notably lower latency (higher performance). This may be due to Java being fundamentally faster than Python, design tricks/shortcuts, or a combination of both. Currently difficult to determine source of improved performance since source code does not appear to be included in the CAIM JAR file. The MatLab version is comparable and often faster for very small datasets. However, Python-CAIM can parallelize discretization, and can thus scale better for datasets with many features.

**CLI Options**

	age: caim.py [-h] [-t TARGET_FIELD] [-o OUTPUT_PATH] [-H] [-q] input_file

	CAIM Algorithm Command Line Tool and Library

	positional arguments:
	input_file            CSV input data file

	optional arguments:
	-h, --help            show this help message and exit
	-t TARGET_FIELD, --target-field TARGET_FIELD
							Target fields as integers (0-indexed) or strings
							corresponding to column names.Negative indices (e.g.
							-1) are allowed.
	-o OUTPUT_PATH, --output-path OUTPUT_PATH
							File path to write discrete form of data in CSV format
	-H, --header          Use first row as column name rows
	-q, --quiet           Minimal information is printed to STDOUT

**Example Usages**

Discretize IRIS data

    python3 ./caim.py datasets/iris.data -t -1 -H

Discretize IRIS data and save discrete results to iris_caim_data.csv

    python3 ./caim.py datasets/iris.data -t -1 -H -o iris_caim.csv

Discretize musk1

    python3 ./caim.py datasets/musk_clean1.csv -t -1

**Interval Output**

Intervals are printed in the form:

    [ 0.13  0.34  0.39  0.66]

Which should be interpretted as:

    [0.13, 0.34](0.34, 0.39](0.39, 0.66]

The output dataset will use the right-end of each interval as the discretized value.

**TODO**

* **Fix Unit Tests**
* Continue to re-implement in Pandas/NumPy for speed (avoid loops)
* Add more test data and corresponding unittests
* Clean-up API and document

[1] Kurgan, L. and Cios, K.J., 2004. CAIM Discretization Algorithm. IEEE Transactions on Knowledge and Data Engineering, 16(2):145-153
