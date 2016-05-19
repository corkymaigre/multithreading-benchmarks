
//-------------------------------------------------------------------------------------------------------------------------
//
//		BENCHMARK 2 - Matrix multiplication
//			| parallel C++ using pthread
//			| using dynamic arrays
//			| dynamic number of threads
//			| each cell in the result matrix is performed by one thread
//
//		Rmq: not using Cell structure as in exercice 3 and 4.
//
//-------------------------------------------------------------------------------------------------------------------------



#pragma region INCLUDES
// INCLUDES ---------------------------------------------------------------------------------------------------------------
#include "stdafx.h"
#include <conio.h>			// using _getch()
#include <ctime>
#include <iomanip>			// using setfill() and setw()
#include <iostream>			// using cout, cin, and cerr
#include <pthread.h>		// using pthread_t
#include <stdio.h>
#include <stdlib.h>
#include <string>			// using string
#include <time.h>
namespace win32
{
#include <windows.h>		// using system("pause"), system("cls") and exit(0)
}
//--------------------------------------------------------------------------------------------------------------------------
#pragma endregion


#pragma region DEFINES
// DEFINES ----------------------------------------------------------------------------------------------------------------
#define APP_TITLE				"Benchmark 2 by Corky Maigre"
#define MENU_ITEM_ENTRY			1
#define MENU_ITEM_CALCUL		2
#define MENU_ITEM_DISPLAY		3
#define MENU_ITEM_EXIT 			4
//-------------------------------------------------------------------------------------------------------------------------
#define MATRIX_SQUARE_SIZE		100
#define MATRIX_LINE_MIN			1
#define MATRIX_LINE_MAX			10
#define MATRIX_COL_MIN			1
#define MATRIX_COL_MAX			10
#define MATRIX_VALUE_MIN		-500
#define MATRIX_VALUE_MAX		500
//-------------------------------------------------------------------------------------------------------------------------
//#define MENU_ON				// if uncommented, a menu appears
//#define DEBUG_ON				// if uncommented, debug is displayed
//#define DISPLAY_MATRIX_ON		// if uncommented, matrices are displayed
#define BENCHMARK_ON			// if uncommented, benchmark is applying
//-------------------------------------------------------------------------------------------------------------------------
#pragma endregion



#pragma region STRUCTURES
// STRUCTURES -------------------------------------------------------------------------------------------------------------
struct Flag
{
	bool is_matrix_allocated = false;
	bool is_matrix_computed = false;
	bool is_matrix_configured = false;
	bool is_matrix_deleted = false;
	bool is_matrix_displayed = false;
	bool is_matrix_generated = false;
	bool is_matrix_initialized = false;
	bool is_matrix_multiplied = false;
	bool is_menu_displayed = false;
	bool is_thread_allocated = false;
	bool is_title_displayed = false;
};

struct Matrix
{
	std::string name;					// name
	int line;							// number of line
	int column;							// number of column
	int cell;							// number of cells
	int** array;						// array with values
};

struct ThreadData						// data given to a thread
{
	int i;
	int j;
};

struct Data								// app data
{
	Matrix *matrix1;					// matrix 1
	Matrix *matrix2;					// matrix 2
	Matrix *matrix3;					// matrix 3
	pthread_t **thread;					// array of threads
	ThreadData **thread_data;			// array of data given to a thread
};
//-------------------------------------------------------------------------------------------------------------------------
#pragma endregion


#pragma region CLASSES
// CLASSES ------------------------------------------------------------------------------------------------------------------
class timer
{
	win32::LARGE_INTEGER start_time_;
public:
	timer()
	{
		QueryPerformanceCounter(&start_time_);
	}
	void restart()
	{
		QueryPerformanceCounter(&start_time_);
	}
	double elapsed() const
	{
		win32::LARGE_INTEGER end_time, frequency;
		QueryPerformanceCounter(&end_time);
		QueryPerformanceFrequency(&frequency);
		return (double(end_time.QuadPart - start_time_.QuadPart)) * 1000 / frequency.QuadPart;
	}
};
//-------------------------------------------------------------------------------------------------------------------------
#pragma endregion


#pragma region FUNCTION DECLARATIONS
// FUNCTION DECLARATIONS --------------------------------------------------------------------------------------------------
/* ERROR */
template <class T>
static void cinCheck(T &var);
/* INITIALIZATION */
static void ConfigureMatrix(Matrix *matrix);
static void AllocateMatrix(Matrix *matrix);
static void GenerateMatrixData(Matrix *matrix);
static void Initialize(Data *data, bool dynamic);
/* COMPUTATION */
static void Multiply(Data *data);
static void Compute(Data *data);
/* DELETING */
static void DeleteMatrix(Matrix *matrix);
static void DeleteMemory(Data *data);
/* DISPLAYING */
static void DisplayTitle(std::string text);
static void DisplayMenu();
static void DisplayMatrix(std::string title, Matrix *matrix);
static void Display(Data *data);
/* THREADING */
static void AllocateThread(Data *data);
static void *MutiplyWithThread(void *arg);
//-------------------------------------------------------------------------------------------------------------------------
#pragma endregion


#pragma region GLOBALS
#ifdef BENCHMARK_ON				// benchmark for speed tests
clock_t mclock;					// benchmark with clock_t
timer mtimer;					// benchmark with Timer class
double elapsed_time;			// elapsed time
#endif // BENCHMARK_ON
Flag *flag = new Flag();		// flags
Data *data = new Data();		// data structure of the application
#pragma endregion


#pragma region MAIN
								// MAIN -------------------------------------------------------------------------------------------------------------------------
int main()
{
	srand(time(0));
	int menu_choice;
	data->matrix1 = new Matrix();
	data->matrix2 = new Matrix();
	data->matrix3 = new Matrix();

#ifndef MENU_ON
	Initialize(data, true);
	Compute(data);
	Display(data);
	DeleteMemory(data);
#else
	{
		do
		{
			DisplayTitle(APP_TITLE);
			DisplayMenu();
			cinCheck(menu_choice);
			system("cls");
			switch (menu_choice)
			{
			case MENU_ITEM_ENTRY:
				Initialize(data, false);
				break;

			case MENU_ITEM_CALCUL:
				Compute(data);
				break;

			case MENU_ITEM_DISPLAY:
				Display(data);
				break;

			case MENU_ITEM_EXIT:
				DeleteMemory(data);
				break;

			default:
				std::cerr << "Incorrect entry, please try again ..." << std::endl << std::endl;
			}
		} while (menu_choice != MENU_ITEM_EXIT);
	}
#endif
	return 0;
}
//-------------------------------------------------------------------------------------------------------------------------
#pragma endregion






// FUNCTIONS DEFINITION ---------------------------------------------------------------------------------------------------
template <class T>
static void cinCheck(T &var)
{
	do
	{
		if (std::cin.fail())
		{
			std::cout << "Incorrect entry, please try again ...";
			std::cin.clear();
			std::cin.sync();
		}
		std::cin >> var;
	} while (std::cin.fail());
}



#pragma region INITIALIZATION
#ifdef MENU_ON
static void ConfigureMatrix(Matrix *matrix)
{
	flag->is_matrix_configured = false;
	int config_choice;
	do
	{
		system("cls");
		std::cout << "Matrix " << matrix->name << std::endl;
		do
		{
			std::cout << "\nNumber of lines " << "(" << MATRIX_LINE_MIN << "-" << MATRIX_LINE_MAX << "): ";
			cinCheck(matrix->line);
		} while ((matrix->line<MATRIX_LINE_MIN) || (matrix->line>MATRIX_LINE_MAX));
		do
		{
			std::cout << "Number of columns " << "(" << MATRIX_COL_MIN << "-" << MATRIX_COL_MAX << "): ";
			cinCheck(matrix->column);
		} while ((matrix->column<MATRIX_COL_MIN) || (matrix->column>MATRIX_COL_MAX));

		system("cls");
		std::cout << "Matrix " << matrix->name << " (" << matrix->line << "x" << matrix->column << ")\n" << std::endl;
		std::cout << "Cancel: [Esc]" << std::endl;
		std::cout << "Confirm: [Espace]" << std::endl;
		config_choice = _getch();

	} while ((config_choice == 27) || (config_choice != 32));
	flag->is_matrix_configured = true;
}
static void GenerateMatrixData(Matrix *matrix)
{
	flag->is_matrix_generated = false;
	system("cls");
	std::cout << "Enter the data\n" << std::endl;
	for (int i = 0; i < matrix->line; i++)
	{
		for (int j = 0; j < matrix->column; j++)
		{
			std::cout << "\n" << matrix->name << " [" << i << "][" << j << "]: ";
			cinCheck(matrix->array[i][j]);
		}
	}
	flag->is_matrix_generated = true;
}
#else
static void ConfigureMatrix(Matrix *matrix)
{
	flag->is_matrix_configured = false;
	matrix->line = MATRIX_SQUARE_SIZE;
	matrix->column = MATRIX_SQUARE_SIZE;
	matrix->cell = matrix->line * matrix->column;
	flag->is_matrix_configured = true;
}
static void GenerateMatrixData(Matrix *matrix)
{
	flag->is_matrix_generated = false;
	for (int i = 0; i < matrix->line; i++)
	{
		for (int j = 0; j < matrix->column; j++)
		{
			int value = 0;
			do
			{
				value = rand();
			} while ((value<MATRIX_VALUE_MIN) || (value>MATRIX_VALUE_MAX));
			matrix->array[i][j] = value;
		}
	}
	flag->is_matrix_generated = true;
}
#endif // MENU_ON


static void AllocateMatrix(Matrix *matrix)
{
	flag->is_matrix_allocated = false;
	matrix->array = new int*[matrix->line];
	for (int i = 0; i<matrix->line; i++)
	{
		matrix->array[i] = new int[matrix->column];
	}
	flag->is_matrix_allocated = true;
}

static void Initialize(Data *data, bool dynamic)
{
	flag->is_matrix_initialized = false;
	data->matrix1->name = "A";
	data->matrix2->name = "B";
	data->matrix3->name = "C";

	if (dynamic)
	{
		ConfigureMatrix(data->matrix1);
		ConfigureMatrix(data->matrix2);
		AllocateMatrix(data->matrix1);
		AllocateMatrix(data->matrix2);
		GenerateMatrixData(data->matrix1);
		GenerateMatrixData(data->matrix2);
	}
	else
	{
		ConfigureMatrix(data->matrix1);
		AllocateMatrix(data->matrix1);
		GenerateMatrixData(data->matrix1);

		ConfigureMatrix(data->matrix2);
		AllocateMatrix(data->matrix2);
		GenerateMatrixData(data->matrix2);
	}
	// Configure and allocate matrix3
	data->matrix3->line = data->matrix1->line;
	data->matrix3->column = data->matrix2->column;
	data->matrix3->cell = data->matrix3->line * data->matrix3->column;
	AllocateMatrix(data->matrix3);
	AllocateThread(data);
	flag->is_matrix_initialized = true;
}
#pragma endregion


#pragma region COMPUTATION
static void *MutiplyWithThread(void *arg)
{
	ThreadData *ldata = (ThreadData*)arg;

	int value = 0;
	for (int k = 0; k<data->matrix1->column; k++)
	{
		value = value + data->matrix1->array[ldata->i][k] * data->matrix2->array[k][ldata->j];
	}

	data->matrix3->array[ldata->i][ldata->j] = value;

#ifdef DEBUG_ON
	std::cout << "\ni = " << ldata->i << std::endl;
	std::cout << "\nj = " << ldata->j << std::endl;
	std::cout << "\nvalue [" << ldata->i << "][" << ldata->j << "] = " << data->matrix3->array[ldata->i][ldata->j];
#endif // DEBUG_ON

	return 0;
}

static void Multiply(Data *data)
{
	if (data->matrix1->column == data->matrix2->line)
	{
		// Create threads
		for (int i = 0; i<data->matrix3->line; i++)
		{
			for (int j = 0; j<data->matrix3->column; j++)
			{
				data->thread_data[i][j].i = i;
				data->thread_data[i][j].j = j;
				pthread_create(&data->thread[i][j], NULL, MutiplyWithThread, (void*)&data->thread_data[i][j]);
			}
		}
		flag->is_matrix_multiplied = true;
	}
	else
	{
		std::cerr << "\nMultiplication impossible, number of lines of the second matrix is different than the number of columns of the first matrix\n" << std::endl;
		system("pause");
		flag->is_matrix_computed = false;
	}
}

static void Compute(Data *data)
{
	if (!flag->is_matrix_initialized)
	{
		std::cerr << "Error: You must enter a matrix !\n" << std::endl;
		system("pause");
	}
	else
	{
#ifdef BENCHMARK_ON
		mclock = clock();								// start the clock
		mtimer.restart();								// start the timer
		Multiply(data);											// do the multiplication for the benchmark
		for (int i = 0; i < data->matrix3->line; i++)
		{
			for (int j = 0; j < data->matrix3->column; j++)
			{
				pthread_join(data->thread[i][j], NULL);
			}
		}
		elapsed_time = mtimer.elapsed();		// stop the timer
		mclock = clock() - mclock;			// stop the clock
#else
		Multiply(data);
#endif // BENCHMARK_ON
	}
}
#pragma endregion


#pragma region DELETING
static void DeleteMatrix(Matrix *matrix)
{
	for (int i = matrix->line - 1; i >= 0; i--)	// destruction from the last allocated element
	{
		delete[] matrix->array[i];
	}
	delete[] matrix->array;
	delete matrix;
}

static void DeleteMemory(Data *data)
{
	if (flag->is_thread_allocated)
	{
		for (int i = data->matrix3->line - 1; i >= 0; i--)
		{
			delete[] data->thread[i];
			delete[] data->thread_data[i];
		}
		delete[] data->thread;
		delete[] data->thread_data;
	}
	if (flag->is_matrix_allocated)	// avoiding the delete of unnalloced memory
	{
		DeleteMatrix(data->matrix3);
		DeleteMatrix(data->matrix2);
		DeleteMatrix(data->matrix1);
	}
	delete[] data;
	delete[] flag;
	exit(0);
}
#pragma endregion


#pragma region DISPLAYING
static void DisplayMenu()
{
	flag->is_menu_displayed = false;
	std::cout << "MENU" << std::endl;
	std::cout << "====\n" << std::endl;
	std::cout << "1. Saisir les matrices" << std::endl;
	std::cout << "2. Multiplier les matrices" << std::endl;
	std::cout << "3. Afficher les matrices" << std::endl;
	std::cout << "4. Quitter" << std::endl << std::endl;
	flag->is_menu_displayed = true;
}

static void DisplayTitle(std::string text)
{
	flag->is_title_displayed = false;
	system("cls");
	int size = text.size();
	std::cout << text << std::endl;
	std::cout << std::setfill('-') << std::setw(size) << "-" << std::endl;
	flag->is_title_displayed = true;
}

static void DisplayMatrix(std::string title, Matrix *matrix)
{
	flag->is_matrix_displayed = false;
	std::cout << title << std::endl << std::endl;
	for (int i = 0; i<matrix->line; i++)
	{
		std::cout << "\t[";
		for (int j = 0; j<matrix->column; j++)
		{
			std::cout << " ";
			std::cout << matrix->array[i][j];
		}
		std::cout << " ]" << std::endl;
	}
	std::cout << std::endl << std::endl;
	flag->is_matrix_displayed = true;
}

static void Display(Data *data)
{
	if (!flag->is_matrix_initialized)
	{
		std::cerr << "Error: You must enter a matrix !\n" << std::endl;
	}
	else
	{
#ifdef DISPLAY_MATRIX_ON
		DisplayMatrix("Matrice A", data->matrix1);
		DisplayMatrix("Matrice B", data->matrix2);
#endif // DISPLAY_MATRIX_ON
		if (!flag->is_matrix_multiplied)
		{
			std::cerr << "Error: Computation has not been done !\n" << std::endl;
		}
		else if (flag->is_matrix_computed)
		{
			std::cerr << "Error: Computation has failed !\n" << std::endl;
		}
		else
		{
#ifdef DISPLAY_MATRIX_ON
			DisplayMatrix("Matrice C (AxB)", data->matrix3);
#endif // DISPLAY_MATRIX_ON

#ifdef BENCHMARK_ON
			std::cout << "Elapsed time with clock :\n\n";
			std::cout << "\t" << (float)mclock / CLOCKS_PER_SEC << " second(s)\n";
			std::cout << "\t" << (float)mclock << " milisecond(s)\n";
			std::cout << "\nElapsed time with timer :\n\n";
			std::cout << "\t" << (float)elapsed_time / 1000 << " second(s)\n";
			std::cout << "\t" << (float)elapsed_time << " milisecond(s)\n\n\n\n";
#endif // BENCHMARK_ON
		}
	}
	system("pause");
}
#pragma endregion



#pragma region THREADING
static void AllocateThread(Data *data)
{
	data->thread = new pthread_t*[data->matrix3->line];
	data->thread_data = new ThreadData*[data->matrix3->line];


	for (int i = 0; i<data->matrix3->line; i++)
	{
		data->thread[i] = new pthread_t[data->matrix3->column];
		data->thread_data[i] = new ThreadData[data->matrix3->column];
	}
}
#pragma endregion






