/*
 * Matrix.h
 *
 *  Created on: Aug 8, 2014
 *      Author: suyog
 *
 *  A template class defining high level matrix operations

 * Modified by vj to just include base container functionality, as needed to 
   store a minibatch. Removes dependency on cuda, platform, random numbers.
 */

#ifndef RUDRA_MATH_MATRIXCONTAINER_H_
#define RUDRA_MATH_MATRIXCONTAINER_H_

#include "rudra/util/defs.h"
#include "rudra/util/Tracing.h"

#include "rudra/util/Checking.h"
#include "rudra/util/Descriptor.h"
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <endian.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <pthread.h>
// vj do not support initialization with random numbers
enum matInit_t{
	_ZEROS,
	_ONES,
	//vj	_RANDN,
	//vj _RANDU,
};

namespace rudra {
template <class T>
class MatrixContainer {

public:
	T* 	     buf; // pointer to the matrix data. Data is of type T
	size_t  dimM; // number of rows
	size_t  dimN; // number of columns
              
	/*  vj       
	Platform platform;
        GPUdata *gpuData;
	*/

	/* constructors */
	MatrixContainer();    // default constructor
	~MatrixContainer();   // default destructor
	MatrixContainer(size_t M_in, size_t N_in); // create an empty matrix
	MatrixContainer(size_t M_in, size_t N_in, T *buf_in); // added on Feb 23, 2015, easier for X10 to pass data around.
	MatrixContainer(size_t M_in, size_t N_in, matInit_t s); // create a new matrix of zeros or ones or randu or randn
	MatrixContainer(const MatrixContainer<T>& mat); // copy constructor for deep copy.

	/* copy constructors */
	template <class U>
		MatrixContainer(const MatrixContainer<U>& mat); // template copy constructor
  
        void defaultInit();

	/* Matrix repetition */
	/*vj	Matrix<T> repMat (size_t M, size_t N) const;
	 */
	/* assignment operators */
	MatrixContainer<T>& operator=(const MatrixContainer<T>& rhs); // assignment operator
	template <class U>
		MatrixContainer<T>& operator=(const MatrixContainer<U>& rhs); // template assignment operator
	/*
        void transfer(const Matrix<T> &fromMatrix,
                      Platform         toPlatform) throw (Exception);
	*/
	//        void randomFill(int seed, float  prob);
	//	        void checkSanity() const throw (Exception);
	//        void checkCompatible(const MatrixContainer<T> &other) const throw (Exception);        

	/*matrix IO*/
	const T& 	operator()  (size_t r, size_t c)	const;
	T& 		operator()  (size_t r, size_t c);
	void 		writeMat	(std::string s) 	const;
	void 		writeBinMat	(std::string s) 	const;

};

MatrixContainer<float> readMat (std::string s);
MatrixContainer<float> readMat (int rpos, int cpos, int r, int c, std::string s); // return a submatrix of size (r x c) starting at position (rpos,cpos)

void	      readMat(float * buf, size_t idx, size_t stride, size_t len, std::string fileName);


template <class T>
MatrixContainer<T>     readBinMat(std::string s);

template <class T>
void	      readBinMat(T * buf, size_t idx, size_t stride, size_t len, std::string fileName);


//==============================
// Constructors, destructors
//==============================

  // Initialize to something consistent
  template<class T>
    void MatrixContainer<T>::defaultInit() {
    buf      = NULL;
    dimM     = 0;
    dimN     = 0;
  }

template<class T>
MatrixContainer<T>::MatrixContainer() {
    defaultInit();
}

template<class T>
MatrixContainer<T>::~MatrixContainer() {
    //printf("[MD %u]%016llX\n",((unsigned int) pthread_self() % 100), buf);
    
    if (buf) {
      //vj      platformCheck(platform);
      //vj      memoryDealloc<T>(buf, platform);
      // copied code from platform/Memory.tcc
      delete[] buf;
      //vj      delete gpuData;
    }

    defaultInit();

}

template<class T>
  MatrixContainer<T>::MatrixContainer(size_t M_in, size_t N_in) {
  //M_in = number of rows
  //N_in = number of cols
                          
  defaultInit();  // make valid in case of error 
  if (M_in == 0 || N_in == 0) {
    std::cout << "Error! Matrix dimension cannot be zero" << std::endl;
    exit(EXIT_FAILURE);
  }
  dimM     = M_in;
  dimN     = N_in; 
  buf      = new T[M_in * N_in];
 }

/*added by Wei on Feb 23, 2015, to provide interop with X10*/
template<class T>
MatrixContainer<T>::MatrixContainer(size_t M_in, size_t N_in, T* buf_in) {
  //vj  RUDRA_CHECK(platform == CPU, "Function is not platform independent");
	dimM = M_in;
	dimN = N_in;
	buf = buf_in;
	//        platform = CPU;
	//        gpuData  = new GPUdata;
}

//vj removed Platform argument
template<class T>
  MatrixContainer<T>::MatrixContainer(size_t M_in, size_t N_in, matInit_t s) {
	//M_in = number of rows
	//N_in = number of cols

  defaultInit();  // in case of failure

	if (M_in == 0 || N_in == 0) {
		std::cout << "Error! Matrix dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}
	dimM     = M_in;
	dimN     = N_in;
        size_t numElements = M_in * N_in;
	buf = new T[numElements];
	switch (s) {

	case _ZEROS:
		// initialize to a matrix of all zeros
		for (size_t i = 0; i < M_in * N_in; ++i) {
			buf[i] = 0;
		}
		break;

	case _ONES:
		// initialize to a matrix of all ones
		for (size_t i = 0; i < M_in * N_in; ++i) {
			buf[i] = 1;
		}
		break;
	default:
		// default case: initialize to all zeros
		for (size_t i = 0; i < M_in * N_in; ++i) {
			buf[i] = 0;
		}
		break;
	}
}

//==============================
// Copy constructor
//==============================
template<class T>
MatrixContainer<T>::MatrixContainer(const MatrixContainer<T>& mat) {

  defaultInit();  // in case of failure
	if (mat.dimM == 0 || mat.dimN == 0) {
		std::cout << "Error! Matrix dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}

	//deallocate existing memory -- not needed for constructor
	/*
	 if(this->buf !=NULL && dimM*dimN !=0){
	 this->~Matrix();
	 }*/

	// allocate new memory
	this->dimM = mat.dimM;
	this->dimN = mat.dimN;
	this->buf      = new T[dimM * dimN];
	// copy data
	memcpy(this->buf, mat.buf, (dimM*dimN)*sizeof(T));
	//vj        memoryCopy(this->buf, this->platform, mat.buf, mat.platform, dimM * dimN);
 }

//==============================
// Template copy constructor
//==============================
template<class T>
template<class U>
MatrixContainer<T>::MatrixContainer(const MatrixContainer<U>& mat) {

  defaultInit();  // in case of failure
	if (mat.dimM == 0 || mat.dimN == 0) {
		std::cout << "Error! Matrix dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}

	//deallocate existing memory -- not needed for the constructor
	/*
	 if(this->buf !=NULL && dimM*dimN !=0){
	 this->~Matrix();
	 }*/

	//allocate new memory
	this->dimM = mat.dimM;
	this->dimN = mat.dimN;
	this->buf      = new T[dimM * dimN];


	//convert (U -> T) and copy data
	for (size_t i = 0; i < dimN * dimM; ++i) {
		this->buf[i] = T(mat.buf[i]);
	}
}

//==============================
// Assignment operator
//==============================
template<class T>
MatrixContainer<T>& MatrixContainer<T>::operator=(const MatrixContainer<T>& rhs) {

  // If self-copy then nothing to do
  if (this != &rhs) {
    size_t numElements = rhs.dimN * rhs.dimM;
                       
    // Check if dimensions match.
    // Only if they do not, then need to reallocate this buf
    if (rhs.dimM != this->dimM ||
        rhs.dimN != this->dimN) {
        
      //deallocate existing memory
      if (this->buf != NULL) {
        this->~MatrixContainer();
      }
      this->dimM     = rhs.dimM;
      this->dimN     = rhs.dimN;
      this->buf      = new T[numElements];
    }
    memcpy(this->buf, rhs.buf, numElements*sizeof(T));
  }
  return (*this);
}

//==============================
// Template assignment operator
//==============================
template<class T>
template<class U>
MatrixContainer<T>& MatrixContainer<T>::operator=(const MatrixContainer<U>& rhs) {
                                                       
	// check if dimensions match
	if (rhs.dimM != this->dimM || rhs.dimN != this->dimN) {
		//	std::cout << "Warning: Dimension mismatch" << std::endl;
		//	exit(EXIT_FAILURE);
	}

	// no need to check for self-copy -- since datatype U needs to be converted to T

	//deallocate existing memory
	if (this->buf != NULL && dimM * dimN != 0) {
		this->~MatrixContainer();
	}
	this->dimM = rhs.dimM;
	this->dimN = rhs.dimN;
	this->buf = new T[dimM * dimN];

	for (size_t i = 0; i < dimN * dimM; ++i) {
		this->buf[i] = T(rhs.buf[i]);
	}
	return (*this);

}

//==============================
// Misc. functions
//==============================

template<class T>
inline T& MatrixContainer<T>::operator ()(size_t r, size_t c) {
	assert(r <= dimM - 1);
	assert(c <= dimN - 1);
	return buf[r * dimN + c];
}

template<class T>
inline const T& MatrixContainer<T>::operator ()(size_t r, size_t c) const {
	assert(r <= dimM - 1);
	assert(c <= dimN - 1);
	return buf[r * dimN + c];
}

template<class T>
void MatrixContainer<T>::writeMat(std::string s) const {
  //    checkSanity();
  
	std::ofstream f1(s.c_str(), std::ios::trunc); //
	if (!f1) {
		std::cout << "Error! failed to open file" << std::endl;
		exit(EXIT_FAILURE);
	} else {
		for (size_t i = 0; i < dimM; ++i) {
			for (size_t j = 0; j < dimN; ++j) {
				f1 << float(buf[j + i * dimN]);
				if (j < dimN - 1)
					f1 << ",";
			}
			f1 << "\n";
		}
	}
	f1.close();
}
template<class T>
void MatrixContainer<T>::writeBinMat (std::string s) const{
	// binary file format : [uint32][uint32][data	]
	//			[rows  ][cols  ][	]	
	std::ofstream f1(s.c_str(), std::ios::out | std::ios::trunc | std::ios::binary); // open in binary mode, discard e
	if(!f1){
		std::cout << "Matrix::writeBinMat::Error! failed to open file: " << s << std::endl;
		exit(EXIT_FAILURE);
	}

	f1.write((char*)&this->dimM,sizeof(uint32));	// write number of rows
	f1.write((char*)&this->dimN,sizeof(uint32));	// write number of cols

	f1.write((char*)this->buf,sizeof(T)*dimM*dimN); // write the buffer

	f1.close();

}
  //===========================
  // read Matrix
  //===========================
  inline MatrixContainer<float> readMat(std::string s) { // added inline on Feb 23, 2015 by Wei so that multiple defs can be avoided
    std::ifstream f1(s.c_str(), std::ios::in); //open file for reading
    if (!f1) {
      std::cout << "Error! failed to open file " << s << " " << std::endl;
      exit(EXIT_FAILURE);
    }

    std::string line;
    std::stringstream ss;
    std::string temp;

    int r = 0;
    int c = 0;

    // figure out the dimensions of the resulting matrix

    // figure out # of rows
    getline(f1, line, '\n');
    while (!f1.eof()) {
      r++;
      getline(f1, line, '\n');
    }

    if (r == 0) {
      std::cout << "Are you trying to read in an empty matrix from " << s<< " ?"<<std::endl;
      exit(EXIT_FAILURE);
    }
    f1.clear();					// clear eof flags
    f1.seekg(0, std::ios::beg); // move the beginning of the file

    //figure out # of columns
    getline(f1, line, '\n');		// read one line.
    ss << line;
    while (!ss.eof()) {
      getline(ss, temp, ',');
      c++;
    }

    // now read in all the elements
    ss.clear();
    f1.clear();					// clear eof flags
    f1.seekg(0, std::ios::beg); // move the beginning of the file

    int ii = 0;
    int jj = 0;
    MatrixContainer<float> ret(r, c, _ZEROS);
    getline(f1, line, '\n');
    while (!f1.eof()) {
      ss << line;
      while (!ss.eof()) {
        getline(ss, temp, ',');
        ret.buf[ii] = std::atof(temp.c_str()); //string to float
        ii++;
      }
      ss.clear();
      jj++;
      getline(f1, line, '\n');
    }

    return ret;
  }
  inline void readMat(float * buf, size_t idx, size_t stride, size_t len, std::string fileName){
    // read a stride # of rows starting at idx. len is the size of each row
    // assuming that size of buffer buf = stride * len * sizeof(float)


    size_t r_t =0; // number of rows parsed so far

    std::ifstream f1(fileName.c_str(), std::ios::in); //open file for reading
    if(!f1){
      std::cout << "Matrix::readMat::Error! failed to open file: " << fileName << std::endl;
      exit(EXIT_FAILURE);
    }

    std::string 		line;
    std::stringstream 	  ss;
    std::string			temp;


    // seek to the row number:rpos
    getline(f1,line,'\n');
    while(!f1.eof() && r_t != idx){
      r_t ++;
      getline(f1,line,'\n');
    }

    if(r_t != idx){
      // we reached end of file.
      std::cout <<"Matrix::readMat::Number of rows in the matrix = " << r_t << std::endl;
      exit(EXIT_FAILURE);
    }

    size_t ii =0; size_t jj = 0;

    while(!f1.eof() && jj < stride){
      ss << line;
      ii = 0;
      while(!ss.eof()){
        getline(ss,temp,',');

        buf[jj*len + ii ] = std::atof(temp.c_str()); //string to float
        ii++;
      }

      ss.clear();
      jj++; 	// keeps track of number of rows parsed so far
      getline(f1,line,'\n');
    }

  };
  inline MatrixContainer<float> readMat(size_t rpos, size_t cpos, size_t r, size_t c, std::string s){
    // read a sub-matrix starting at (rpos,cpos). The size of the sub-matrix is r x c
    // note: minimum value of rpos and cpos = 0


    MatrixContainer<float> res(r,c,_ZEROS);

    size_t r_t =0; // number of rows parsed so far

    std::ifstream f1(s.c_str(), std::ios::in); //open file for reading
    if(!f1){
      std::cout << "Matrix::readMat::Error! failed to open file: " << s << std::endl;
      exit(EXIT_FAILURE);
    }

    std::string 		line;
    std::stringstream 	  ss;
    std::string		temp;


    // seek to the row number:rpos
    getline(f1,line,'\n');
    while(!f1.eof() && r_t != rpos){

      getline(f1,line,'\n');
      ++r_t;
    }

    if(r_t != rpos){
      // we reached end of file.
      std::cout <<"Matrix::readMat::Number of rows in the matrix = " << r_t << std::endl;
      exit(EXIT_FAILURE);
    }

    size_t ii =0; size_t jj = 0;


    while(!f1.eof() && jj < r){
      ss << line;
      ii = 0;
      while(!ss.eof()){
        getline(ss,temp,',');

        if(ii > cpos -1 && ii < cpos + c ){
          res.buf[jj*c + ii - cpos] = std::atof(temp.c_str()); //string to float

        }
        ii++;
      }
      ss.clear();
      jj++; 	// keeps track of number of rows parsed so far
      getline(f1,line,'\n');
    }
    return res;
  }
  template <class T>
    MatrixContainer<T> readBinMat (std::string s) {
    std::ifstream f1(s.c_str(), std::ios::in | std::ios::binary);
    if(!f1){
      std::cout << "Matrix::readBinMat::Error! failed to open file: " << s << std::endl;
      exit(EXIT_FAILURE);
    }
    int r,c;

    f1.read((char*)&r,sizeof(uint32)); 	// read number of rows
    f1.read((char*)&c,sizeof(uint32));	// read number of cols
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // swap byte order
    r = be32toh(r);
    c = be32toh(c);
#endif

    if(r < 0 || c < 0){
      std::cout << "Matrix::readBinMat::Invalid matrix dimensions:" << r << " X " << c << std::endl;
      exit(EXIT_FAILURE);		

    }
		
    MatrixContainer<T> res(r,c,_ZEROS);
    f1.read((char*)res.buf,sizeof(T)*r*c);

    switch (sizeof(T)){

    case 1:
      // no need to swap byte order
      // do nothing
      break;
    case 2:
      // 16 bit data -- need to swap byte order
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      // swap byte order
      for (size_t i=0; i<r*c; i++) {
        uint16 swapped = be16toh(*reinterpret_cast<uint16 *>(&res.buf[i]));
        res.buf[i] = *reinterpret_cast<uint16 *>(&swapped);
      }
      break;
#endif
	
    case 4:	
      // 32 bit data -- need to swap byte order
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      // swap byte order
      for (size_t i=0; i<r*c; i++) {
        uint32 swapped = be32toh(*reinterpret_cast<uint32 *>(&res.buf[i]));
        res.buf[i] = *reinterpret_cast<float *>(&swapped);
      }
      break;
#endif
	
    default:
      break;
	

    }
    return res;
  }
  template <class T>
    inline void readBinMat (T * buf, size_t idx, size_t stride, size_t len, std::string fileName){
    // stride = number of rows to be read
    // len = col length to be read

    std::ifstream f1(fileName.c_str(), std::ios::in | std::ios::binary); //open file for reading
    if(!f1){
      std::cout << "MatrixContainer::readBinMat::Error! failed to open file: " << fileName << std::endl;
      exit(EXIT_FAILURE);
    }

    uint32 r,c;

    f1.read((char*)&r,sizeof(uint32)); 	// read number of rows
    f1.read((char*)&c,sizeof(uint32));	// read number of cols
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // swap byte order
    r = be32toh(r);
    c = be32toh(c);
#endif
	
    size_t currPos = f1.tellg();		// bytes read so far
    size_t seekPos = currPos + idx * len * sizeof(T);

    //sanity checks:
    if(seekPos + stride*len*sizeof(T) > currPos + size_t(r) * size_t(c) * sizeof(T)){
      // Trying to read more than what the file contains
      std::cout <<"MatrixContainer::readBinMat::Read exceeds file dimensions: r = " << r <<" c = " << c<< std::endl;
      exit(EXIT_FAILURE);
    }

    f1.seekg(seekPos);
    f1.read((char*)buf,stride*len*sizeof(T));

    switch (sizeof(T)){

    case 1:
      // no need to swap byte order
      // do nothing
      break;
    case 2:
      // 16 bit data -- need to swap byte order
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      // swap byte order
      for (size_t i=0; i<stride*len; i++) {
        uint16 swapped = be16toh(*reinterpret_cast<uint16 *>(&buf[i]));
        buf[i] = *reinterpret_cast<uint16 *>(&swapped);
      }
      break;
#endif

    case 4:
      // 32 bit data -- need to swap byte order
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      // swap byte order
      for (size_t i=0; i<stride*len; i++) {
        uint32 swapped = be32toh(*reinterpret_cast<uint32 *>(&buf[i]));
        buf[i] = *reinterpret_cast<float *>(&swapped);
      }
      break;
#endif

    default:
      break;


    }
  }

 } /* namespace rudra */
#endif /* RUDRA_MATH_MATRIXCONTAINER_H_ */
