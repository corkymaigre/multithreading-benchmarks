{{{

Multithreading Benchmarks
=========================


Here's the stuff.

- [Homepage](https://github.com/CorkyMaigre)
- [Source files](https://github.com/CorkyMaigre/multithreading-benchmarks)
- [Website](http://www.corkymaigre.be/)


Table of contents
=================

- [HEHLan](#HehLan)
- [Table of contents](#table-of-contents)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
	- [Benchmark 1](#benchmark-1)
	- [Benchmark 2](#benchmark-2)
	- [Benchmark 3](#benchmark-3)
	- [Benchmark 4](#benchmark-4)
- [Contribute](#contribute)
- [Bugs](#bugs)



Configuration
=============

First of all, you have to set that we use pthread library by typing 'pthreadVC2.lib' into the additional dependencies found at
'Property' > 'Links Editor' > 'Additional Dependencies' as shown on the picture below.
<center>
	<img src="assets/img/config-000.png" width="500px" />
</center>

>**CAUTION**
> It is possible that you need to type 'HAVE_STRUCT_TIMESPEC' into the preprocessor definition found at
>'Property' > 'C/C++' > 'Preprocessor' > 'Preprocessor Definition' as shown on the picture below.
<center>
	<img src="assets/img/config-001.png" width="500px" />
<\center>



Benchmarks
==========

All benchmarks presented here are resulted from one console application consisting of doing a matrix multiplication.
Each benchmark is computed for a specific use of threading.

Benchmark 1
-----------

This first benchmark is created by executing the sequential code in C++. The code uses dynamic arrays and a 'Matrix' structure.

Number of cells | Elapsed Time
----------------|------------
100				| 0.0000 ms
200				| 0.0000 ms
500				| 0.0000 ms
1000			| 0.0000 ms


Benchmark 2
-----------

The second benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads. Each cell in the result matrix is performed by one thread.

>**NOTE**
> Not using the 'Cell' structure as in the benchmarks 3 and 4.



Benchmark 3
-----------

The third benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads. Each row in the result matrix is performed by one thread.

>**NOTE**
> Using the 'Cell' structure.


Benchmark 4
-----------

The fourth benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads.
Cells in the result matrix are performed by one of the threads according to a shifting algorithm.



Contribute
==========

No one has contribute to the project.



Bugs
====

No bugs encountered yet.


}}}