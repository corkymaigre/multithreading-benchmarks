// benchmark-1.cpp : définit le point d'entrée pour l'application console.
//


//-------------------------------------------------------------------------------------------------------------------------
//
//		BENCHMARK 1 - v1
//			| base for others exercices
//			| sequential C++
//			| matrix multiplication
//			| dynamic arrays
//		
//		--> Using Matrix structure and references
//
//
//-------------------------------------------------------------------------------------------------------------------------


// INCLUDES ---------------------------------------------------------------------------------------------------------------
#include "stdafx.h"
#include <conio.h>			// using _getch()
#include <ctime>
#include <iomanip>			// using setfill() and setw()
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
namespace win32
{
#include <windows.h>		// using system("pause"), system("cls") and exit(0)
}
//-------------------------------------------------------------------------------------------------------------------------


// DEFINES ----------------------------------------------------------------------------------------------------------------
#define APP_TITLE				"Exercice 1 by Corky Maigre"
#define ENTRY					1
#define CALCUL					2
#define DISPLAY					3
#define EXIT 					4
#define MATRIX_SQUARE_SIZE		2
#define MATRIX_LINE_MIN			1
#define MATRIX_LINE_MAX			10
#define MATRIX_COL_MIN			1
#define MATRIX_COL_MAX			10
#define MATRIX_VALUE_MIN		-500
#define MATRIX_VALUE_MAX		500
#define AUTO							// If uncommented, computation is doing randomly
//-------------------------------------------------------------------------------------------------------------------------


// STRUCTURES -------------------------------------------------------------------------------------------------------------
struct Flag
{
	bool entry = false;
	bool multiply = false;
	bool compute_failed = false;
};

struct Matrix
{
	std::string name;		// name
	int line;				// number of line
	int column;				// number of column
	int** array;			// array
};
//-------------------------------------------------------------------------------------------------------------------------

// CLASS ------------------------------------------------------------------------------------------------------------------
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

// FUNCTIONS DECLARATION --------------------------------------------------------------------------------------------------
/* ERROR */
template <class T>
static void cinCheck(T &var);
/* INITIALIZATION */
static void ConfigureMatrix(Matrix &matrix1, Matrix &matrix2);
static void ConfigureMatrix(Matrix &matrix);
static void AllocateMatrix(Matrix &matrix);
static void GenerateMatrixData(Matrix &matrix, bool dynamic);
static void Initialize(Matrix &matrix1, Matrix &matrix2, bool dynamic, Flag &flag);
/* COMPUTATION */
static void Multiply(Matrix matrix1, Matrix matrix2, Matrix &matrix3, Flag &flag);
static void Compute(Matrix matrix1, Matrix matrix2, Matrix &matrix3, Flag &flag);
/* DELETING */
static void DeleteMatrix(Matrix &matrix);
static void DeleteMemory(Matrix &matrix1, Matrix &matrix2, Matrix &matrix3, Flag flag);
/* DISPLAYING */
static void DisplayTitle(std::string text);
static void DisplayMenu();
static void DisplayMatrix(std::string title, Matrix matrix);
static void Display(Matrix A, Matrix B, Matrix C, Flag flag);
//-------------------------------------------------------------------------------------------------------------------------

clock_t benchmark;
double elapsed_time;

// MAIN -------------------------------------------------------------------------------------------------------------------------
int main()
{
	int menu_choice;
	Flag flag;					// All flags
	Matrix A;					// Matrix A
	Matrix B;					// Matrix B
	Matrix C;					// Matrix C = A * B

#ifdef AUTO
	Initialize(A, B, true, flag);
	Compute(A, B, C, flag);
	Display(A, B, C, flag);
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
			case ENTRY:
				Initialize(A, B, false, flag);
				break;

			case CALCUL:
				Compute(A, B, C, flag);
				break;

			case DISPLAY:
				Display(A, B, C, flag);
				break;

			case EXIT:
				DeleteMemory(A, B, C, flag);
				break;

			default:
				std::cerr << "Incorrect entry, please try again ..." << std::endl << std::endl;
			}
		} while (menu_choice != EXIT);
	}
#endif
	return 0;
}
//-------------------------------------------------------------------------------------------------------------------------


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
static void ConfigureMatrix(Matrix &matrix1, Matrix &matrix2)
{
	// This code below randomly defines the matrix size, so it is not appropriate for benchmarks
	/*srand(time(0));
	do
	{
	matrix1.line = rand() % (Nmax - Nmin) + Nmax;
	matrix1.column = rand() % (Nmax - Nmin) + Nmax;
	matrix2.line = matrix1.column;
	matrix2.column = rand() % (Nmax - Nmin) + Nmax;
	} while ((matrix1.line<Nmin) || (matrix1.line>Nmax) || (matrix2.line<Nmin) || (matrix2.line>Nmax));*/

	// This code below is appropriate for benchmarks
	matrix1.line = MATRIX_SQUARE_SIZE;
	matrix1.column = MATRIX_SQUARE_SIZE;
	matrix2.line = MATRIX_SQUARE_SIZE;
	matrix2.column = MATRIX_SQUARE_SIZE;
}

static void ConfigureMatrix(Matrix &matrix)
{
	int config_choice;
	do
	{
		system("cls");
		std::cout << "Matrix " << matrix.name << std::endl;
		do
		{
			std::cout << "\nNumber of lines " << "(" << MATRIX_LINE_MIN << "-" << MATRIX_LINE_MAX << "): ";
			cinCheck(matrix.line);
		} while ((matrix.line<MATRIX_LINE_MIN) || (matrix.line>MATRIX_LINE_MAX));
		do
		{
			std::cout << "Number of columns " << "(" << MATRIX_COL_MIN << "-" << MATRIX_COL_MAX << "): ";
			cinCheck(matrix.column);
		} while ((matrix.column<MATRIX_COL_MIN) || (matrix.column>MATRIX_COL_MAX));

		system("cls");
		std::cout << "Matrix " << matrix.name << " (" << matrix.line << "x" << matrix.column << ")\n" << std::endl;
		std::cout << "Cancel: [Esc]" << std::endl;
		std::cout << "Confirm: [Espace]" << std::endl;
		config_choice = _getch();

	} while ((config_choice == 27) || (config_choice != 32));
}

static void AllocateMatrix(Matrix &matrix)
{
	matrix.array = new int*[matrix.line];
	for (int i = 0; i<matrix.line; i++)
	{
		matrix.array[i] = new int[matrix.column];
	}
}

static void GenerateMatrixData(Matrix &matrix, bool dynamic)
{
	if (dynamic)
	{
		srand(time(NULL));
		for (int i = 0; i < matrix.line; i++)
		{
			for (int j = 0; j < matrix.column; j++)
			{
				int value = 0;
				do
				{
					value = rand();
				} while ((value<MATRIX_VALUE_MIN) || (value>MATRIX_VALUE_MAX));
				matrix.array[i][j] = value;
			}
		}
	}
	else
	{
		system("cls");
		std::cout << "Enter the data\n" << std::endl;
		for (int i = 0; i < matrix.line; i++)
		{
			for (int j = 0; j < matrix.column; j++)
			{
				std::cout << "\n" << matrix.name << " [" << i << "][" << j << "]: ";
				cinCheck(matrix.array[i][j]);
			}
		}
	}
}

static void Initialize(Matrix &matrix1, Matrix &matrix2, bool dynamic, Flag &flag)
{
	matrix1.name = "A";
	matrix2.name = "B";
	if (dynamic)
	{
		ConfigureMatrix(matrix1, matrix2);
		AllocateMatrix(matrix1);
		AllocateMatrix(matrix2);
		GenerateMatrixData(matrix1, dynamic);
		GenerateMatrixData(matrix2, dynamic);
	}
	else
	{
		ConfigureMatrix(matrix1);
		AllocateMatrix(matrix1);
		GenerateMatrixData(matrix1, dynamic);
		ConfigureMatrix(matrix2);
		AllocateMatrix(matrix2);
		GenerateMatrixData(matrix2, dynamic);
	}
	flag.entry = true;
}
#pragma endregion


#pragma region COMPUTATION
static void Multiply(Matrix matrix1, Matrix matrix2, Matrix &matrix3, Flag &flag)
{
	// Configure and allocate matrix3
	matrix3.line = matrix1.line;
	matrix3.column = matrix2.column;
	AllocateMatrix(matrix3);

	if (matrix1.column == matrix2.line)
	{
		for (int i = 0; i<matrix3.line; i++)
		{
			for (int j = 0; j<matrix3.column; j++)
			{
				int value = 0;
				for (int k = 0; k<matrix1.column; k++)
				{
					value = value + (matrix1.array[i][k] * matrix2.array[k][j]);
				}
				matrix3.array[i][j] = value;
			}
		}
		flag.multiply = true;
	}
	else
	{
		std::cerr << "\nMultiplication impossible, number of lines of the second matrix is different than the number of columns of the first matrix\n" << std::endl;
		system("pause");
		flag.compute_failed = true;
	}
}

static void Compute(Matrix matrix1, Matrix matrix2, Matrix &matrix3, Flag &flag)
{
	if (!flag.entry)
	{
		std::cerr << "Error: You must enter a matrix !\n" << std::endl;
		system("pause");
	}
	else
	{
		benchmark = clock();							// Start the clock
		Multiply(matrix1, matrix2, matrix3, flag);
		benchmark = clock() - benchmark;				// Stop the clock		
	}
}
#pragma endregion


#pragma region DELETING
static void DeleteMatrix(Matrix &matrix)
{
	for (int i = matrix.line - 1; i >= 0; i--)	// destruction from the last allocated element
	{
		delete[] matrix.array[i];
	}
	delete[] matrix.array;
}

static void DeleteMemory(Matrix &matrix1, Matrix &matrix2, Matrix &matrix3, Flag flag)
{
	if (flag.entry)	// avoiding the delete of unnalloced memory
	{
		DeleteMatrix(matrix3);
		DeleteMatrix(matrix2);
		DeleteMatrix(matrix1);
	}
	exit(0);
}
#pragma endregion


#pragma region DISPLAYING
static void DisplayMenu()
{
	std::cout << "MENU" << std::endl;
	std::cout << "====\n" << std::endl;
	std::cout << "1. Saisir les matrices" << std::endl;
	std::cout << "2. Multiplier les matrices" << std::endl;
	std::cout << "3. Afficher les matrices" << std::endl;
	std::cout << "4. Quitter" << std::endl << std::endl;
}

static void DisplayTitle(std::string text)
{
	system("cls");
	int size = text.size();
	std::cout << text << std::endl;
	std::cout << std::setfill('-') << std::setw(size) << "-" << std::endl;
}

static void DisplayMatrix(std::string title, Matrix matrix)
{
	std::cout << title << std::endl << std::endl;
	for (int i = 0; i<matrix.line; i++)
	{
		std::cout << "\t[";
		for (int j = 0; j<matrix.column; j++)
		{
			std::cout << " ";
			std::cout << matrix.array[i][j];
		}
		std::cout << " ]" << std::endl;
	}
	std::cout << std::endl << std::endl;
}

static void Display(Matrix A, Matrix B, Matrix C, Flag flag)
{
	if (!flag.entry)
	{
		std::cerr << "Error: You must enter a matrix !\n" << std::endl;
	}
	else
	{
		DisplayMatrix("Matrice A", A);
		DisplayMatrix("Matrice B", B);
		if (!flag.multiply)
		{
			std::cerr << "Error: Computation has not been done !\n" << std::endl;
		}
		else if (flag.compute_failed)
		{
			std::cerr << "Error: Computation has failed !\n" << std::endl;
		}
		else
		{
			DisplayMatrix("Matrice C (AxB)", C);
			std::cout << "Elapsed time : " << (float)benchmark / CLOCKS_PER_SEC << " second(s)" << std::endl;
			std::cout << "Elapsed time : " << (float)benchmark << " milisecond(s)" << std::endl;
			//std::cout << "Elapsed time : " << (float) elapsed_time << " milisecond(s)" << std::endl;
		}
	}
	system("pause");
}
#pragma endregion
//-------------------------------------------------------------------------------------------------------------------------


