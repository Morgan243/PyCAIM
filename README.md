CAIM is a supervised discretization method [1] and Python-CAIM is a Python implementation of CAIM. **This is a work in progres, results should be closely inspected**. Pull requests welcome.

There is a [MATLAB implementation](http://www.mathworks.com/matlabcentral/fileexchange/24344-caim-discretization-algorithm) by Guangdi Li and a [Java implementation](http://www.cioslab.vcu.edu/index.html) (Research->Data Mining Toole) by the author.

Current Python-CAIM is working on UCI's Musk1 dataset as well as other toy datasets. Results are validated against the Java implementation (see above). 

On performance, the Java implementation has notably lower latency (higher performance). This may be due to Java being fundamentally faster than Python, design tricks/shortcuts, or a combination of both. Currently difficult to determine source of improved performance since source code does not appear to be included in the CAIM JAR file. The MatLab version is comparable and often faster for very small datasets. However, Python-CAIM can parallelize discretization, and can thus scale better for datasets with many features.

**TODO**

* Continue to re-implement in Pandas/NumPy for speed (avoid loops)
* Add more test data and corresponding unittests
* Clean-up API and document

[1] Kurgan, L. and Cios, K.J., 2004. CAIM Discretization Algorithm. IEEE Transactions on Knowledge and Data Engineering, 16(2):145-153
