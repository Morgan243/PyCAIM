Algo. is primarily based on the [MATLAB implementation](http://www.mathworks.com/matlabcentral/fileexchange/24344-caim-discretization-algorithm)

Current PyCAIM implementation is working on the test data used with the MATLAB implementation. However, the MATLAB implementation is at least twice as fast when comapared to PyCAIM run as a unittest (no CLI arg parsing/building).

**TODO**

* Continue to re-implement in Pandas/NumPy for speed (avoid loops)
* Add more test data and corresponding unittests
* Clean-up API and document
