Multithreading Benchmarks
=========================

This project intended to compare the performance of a console application using CPU & GPU programming.
The console application consisted of doing a square matrix multiplication via several methods :
- CPU Programming
	- Sequential code
	- Multithreading code (one thread per cell of the result matrix)
	- Multithreading code (one thread per row of the result matrix)
	- Multithreading code (each cell of the result matrix is assigned to one of the *n* threads)
- GPU Programming
Each programming method was compared each other by doing a benchmark on the elapsed time for the computation.

My personal information :
- [Homepage](https://github.com/CorkyMaigre)
- [Source files](https://github.com/CorkyMaigre/multithreading-benchmarks)
- [Website](http://www.corkymaigre.be/)


Table of contents
=================

- [Introduction](#introduction)
- [Table of contents](#table-of-contents)
- [Configuration](#configuration)
	- [Hardware Specifications](#hardware-specifications)
	- [IDE Configuration](#ide-configuration)
- [Benchmarks](#benchmarks)
	- [Benchmark 1](#benchmark-1)
	- [Benchmark 2](#benchmark-2)
	- [Benchmark 3](#benchmark-3)
	- [Benchmark 4](#benchmark-4)
- [Conclusion](#conclusion)
- [Contribute](#contribute)
- [Bugs](#bugs)


Introduction
============




Configuration
=============

Hardware Specifications
-----------------------

Using an ASUS X93S Series laptop.

Charasteristics | Description
----------------|------------
Processor		| Intel Core i5 2430M 2.4 GHz ~ 2.9 GHz
Operating System| Windows 7 Home Premium
Chipset			| Intel HM65 Express
Memory			| DDR3 1333 MHz SDRAM, 4096 MB, (1 x 4096 MB)
Display			| 18.4" 16:9 Full HD (1920x1080) LED Backlight
Graphic			| NVIDIA® GeForce® GT 540M with 1GB DDR3 VRAM
Storage			| 1 TB 7200 rpm
Optical Drive	| DVD player
Card Reader		| Card reader ( SD/ SDHC/ MS/ MS Pro/ MMC)
Webcam			| 0.3 Mega Pixel Fixed web camera
Networking		| Integrated 802.11 b/g/n, Bluetooth™ V2.1+EDR, 10/100/1000 Base T
Interface		| 1 x Microphone-in jack, 1 x Headphone-out jack, 1 x VGA port / Mini D-sub 15 pins for external monitor, 1 x USB 3.0 port, 3 x USB 2.0 ports, 1 x RJ45 LAN Jack for LAN insert, 1 x HDMI
Audio			| Built-in Speakers And Microphone, SonicFocus, Altec Lansing® Speakers
Battery			| 6Cells : 5200 mAh 56 Whrs
Power Adapter	| Output : 19 V DC, 6.3 A, 120 W Input : 100 -240 V AC, 50/60 Hz universal
Dimensions		| 44.1 x 29.5 x 4.23 ~5.59 cm (WxDxH)
Weight			| 4.11 kg (with 6 cell battery)
Note			| Master HDD: 3.5” SATA, Second HDD: 2.5” SATA


IDE Configuration
-----------------

The IDE used is Visual Studio Community 2015.

First of all, you have to set that we use pthread library by typing 'pthreadVC2.lib' into the additional dependencies found at
'Property' > 'Links Editor' > 'Additional Dependencies' as shown on the picture below.
<img src="assets/img/config-000.png" width="600px" />

>**CAUTION**
> It is possible that you need to type 'HAVE_STRUCT_TIMESPEC' into the preprocessor definition found at
>'Property' > 'C/C++' > 'Preprocessor' > 'Preprocessor Definition' as shown on the picture below.
> Otherwise you will have this error message: Error C2011 'timespec' : redefinition of type 'struct'
<img src="assets/img/config-001.png" width="600px" />



Benchmarks
==========

All benchmarks presented here are resulted from one console application consisting of doing a matrix multiplication.
Each benchmark is computed for a specific use of threading.

Benchmark 1
-----------

This first benchmark is created by executing the sequential code in C++. The code uses dynamic arrays and a 'Matrix' structure.

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 0.000000 s   | 0.000000 s   | 0.000000 s   | 0.000000 s    | 0.000000 s   |
25	x 25		|625			| 0.000144 s   | 0.000106 s   | 0.000198 s   | 0.000108 s    | 0.000106 s   |
50	x 50		|2,500			| 0.001092 s   | 0.001143 s   | 0.001261 s   | 0.000897 s    | 0.000845 s   |
100 x 100		|10,000			| 0.007243 s   | 0.009130 s   | 0.010531 s   | 0.012686 s    | 0.006135 s   |
200 x 200		|40,000			| 0.069280 s   | 0.050119 s   | 0.096451 s   | 0.070126 s    | 0.080317 s   |  
500 x 500		|250,000		| 0.908748 s   | 0.976781 s   | 0.917461 s   | 0.906997 s    | 0.935186 s   |
1,000 x 1,000	|1,000,000		| 14.82270 s   | 14.90280 s   | 15.32160 s   | 15.16810 s    | 15.07080 s   |
1,500 x 1,500	|2,250,000		| 51.84170 s   | 53.78350 s   | 61.38590 s   | 57.83540 s    | 57.28700 s   |
2,000 x 2,000	|4,000,000		| 144.4250 s   | 144.3840 s   | 138.8950 s   | 136.5640 s    | 143.8150 s   |




Benchmark 2
-----------

The second benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads. Each cell in the result matrix is performed by one thread.

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 0.001818 s   | 0.000263 s   | 0.001539 s   | 0.005473 s    | 0.001581 s   |
25	x 25		|625			| 0.651238 s   | 0.641366 s   | 0.383544 s   | 0.528443 s    | 0.590903 s   |
50	x 50		|2,500			| 1.956470 s   | 2.183280 s   | 5.004200 s   | 2.249170 s    | 1.905320 s   |
100 x 100		|10,000			| 8.355190 s   | 8.410570 s   | 7.513020 s   | 7.785250 s    | 7.876830 s   |
200 x 200		|40,000			| 32.00290 s   | 33.54810 s   | 34.70680 s   | 31.60480 s    | 32.82290 s   |  
500 x 500		|250,000		| 210.6790 s   | 212.8600 s   | 216.6650 s   | 217.6900 s    | 221.4850 s   |
1,000 x 1,000	|1,000,000		| 958.3180 s   | 918.7720 s   | 931.3840 s   | 905.0450 s    | 1000.510 s   |
1,500 x 1,500	|2,250,000		| 0.000000 s   | 0.000000 s   | 0.000000 s   | 0.000000 s    | 0.000000 s   |
2,000 x 2,000	|4,000,000		| 0.000000 s   | 0.000000 s   | 0.000000 s   | 0.000000 s    | 0.000000 s   |

Before launching the console application for a square matrix of 1,000 x 1,000, the memory is low (but Visual Studio takes a big part of memory).

<img src="assets/img/bench-2-1000-before.png" width="800px" />


At the end of the console application for a square matrix of 1,000 x 1,000, the memory is full because all pointers take a lot of space and there are 1,000,000 of threads in memory.

<img src="assets/img/bench-2-1000-end.png" width="800px" />

When the application finished, all the memory is released.

<img src="assets/img/bench-2-1000-after.png" width="800px" />


>**NOTE**
> Not using the 'Cell' structure as in the benchmarks 3 and 4.



Benchmark 3
-----------

The third benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads. Each row in the result matrix is performed by one thread.

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 0.000655 s   | 0.000203 s   | 0.000496 s   | 0.000171 s    | 0.000187 s   |
25	x 25		|625			| 0.010548 s   | 0.011697 s   | 0.016335 s   | 0.021629 s    | 0.017597 s   |
50	x 50		|2,500			| 0.022197 s   | 0.023205 s   | 0.029736 s   | 0.024637 s    | 0.028992 s   |
100 x 100		|10,000			| 0.053766 s   | 0.074381 s   | 0.043908 s   | 0.066718 s	 | 0.059816 s   |
200 x 200		|40,000			| 0.118335 s   | 0.112993 s   | 0.104711 s   | 0.165889 s	 | 0.254931 s   |  
500 x 500		|1,000,000		| 0.708654 s   | 0.537136 s   | 0.569650 s   | 0.527360 s    | 0.493854 s   |
1,500 x 1,500	|2,250,000		| 25.71800 s   | 5.512310 s   | 3.487280 s   | 23.34210 s    | 26.98990 s   |
2,000 x 2,000	|4,000,000		| 51.79730 s   | 32.06580 s   | 50.01410 s   | 60.21770 s    | 53.90000 s   |

I cannot explain the difference between elapsed 1 and 2 for a matrix 1,500 x 1,500.

In the figure below you can see the performance of the computer when the mutiplication of two 2000 x 2000 matrix is performed.

<img src="assets/img/bench-3-2000-computation.png" width="800px" />

In the figure below you can see that the memory empties at the end of the program and then all memory is free when the executable is closed.

<img src="assets/img/bench-3-2000-after.png" width="800px" />


>**NOTE**
> Using the 'Cell' structure.


Benchmark 4
-----------

The fourth benchmark is created by executing the parallel code in C++ using pthread.
The code uses dynamic arrays and a dynamic number of threads.
Cells in the result matrix are performed by one of the threads according to a shifting algorithm.

### Number of threads: 2

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 0.535821 ms  | 1.885420 ms  | 0.584571 ms  | 0.173618 ms   | 1.522790 ms  |
25	x 25		|625			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
50	x 50		|2,500			| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
100 x 100		|10,000			| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms	 | 0.000000 ms  |
200 x 200		|40,000			| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms	 | 0.000000 ms  |  
500 x 500		|1,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
1,500 x 1,500	|2,250,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
2,000 x 2,000	|4,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |

Error: the result matrix is not correct with matrix 3x3 and plus.


### Number of threads: 3

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 3.163180 ms  | 1.684440 ms  | 0.687203 ms  | 0.855262 ms   | 1.056250 ms  |
25	x 25		|625			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
50	x 50		|2,500			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
100 x 100		|10,000			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
200 x 200		|40,000			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       | 
500 x 500		|1,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
1,500 x 1,500	|2,250,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
2,000 x 2,000	|4,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |

Error: the result matrix is not correct.


### Number of threads: 4

Matrix Dimension|Number of cells|Elapsed Time 1|Elapsed Time 2|Elapsed Time 3| Elapsed Time 4|Elapsed Time 5|
----------------|---------------|--------------|--------------|--------------|---------------|--------------|
2	x 2			|4				| 1.236710 ms  | 0.812498 ms  | 2.145420 ms  | 0.855262 ms   | 4.672720 ms  |
25	x 25		|625			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
50	x 50		|2,500			|  ERROR       |  ERROR       |  ERROR       |  ERROR        |  ERROR       |
100 x 100		|10,000			| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
200 x 200		|40,000			| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  | 
500 x 500		|1,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
1,500 x 1,500	|2,250,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |
2,000 x 2,000	|4,000,000		| 0.000000 ms  | 0.000000 ms  | 0.000000 ms  | 0.000000 ms   | 0.000000 ms  |

Error: the result matrix is not correct.


>**NOTE**
> Using the 'Cell' structure.


Conclusion
==========

No conclusion yet


Contribute
==========

No one has contribute to the project.



Bugs
====

- Error with the memory deleting:
	- detected the 01 May 2016
	- solved the 01 May 2016
- Error on benchmark 4 with matrix n x n where n > 2
	- detected the 30 April 2016


