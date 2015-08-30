/*
 * Tensor.h
 *
 *  Created on: Nov 25, 2014
 *      Author: suyog
 *
 *      This is a container class : a matrix of same sized matrices
 *      --essentially implements a 4-dimensional datastructure containing elements of type T
 * vj: Moved from rudra::math to rudra::misc, changed to reference MatrixContainer instead of Matrix.
 */

#ifndef RUDRA_MATH_TENSOR_H_
#define RUDRA_MATH_TENSOR_H_
#include "rudra/misc/MatrixContainer.h"

namespace rudra {
template <class T>
class Tensor {
public:

	size_t 		dimM;	// number of rows in a matrix
	size_t 		dimN;	// number of cols in a matrix
	size_t 		dimK;	// number of rows of matrices
	size_t 		dimP;	// number of cols of matrices
	MatrixContainer<T> * buf;	// array of matrices, always on CPU; individual matrices on any device

	Tensor();
	Tensor(size_t M_in, size_t N_in, size_t K_in, size_t P_in, Platform platform_in); 	        // create an empty 4D-tensor
	Tensor(size_t M_in, size_t N_in, size_t K_in, size_t P_in, matInit_t s, Platform platform_in); 	// create a new 4D-tensor of zeros or ones or randu or randn

	/* copy constructors */
	Tensor(const Matrix<T>& mat); 								// copy constructor for deep copy.
	Tensor(const Tensor<T>& mat); 								// copy constructor for deep copy.
	template <class U>
		Tensor(const Tensor<U>& mat); 							// template copy constructor

	/* assignment operators */
	Tensor<T>& operator=(const Tensor<T>& rhs); 				// assignment operator
	template <class U>
		Tensor<T>& operator=(const Tensor<U>& rhs); 			// template assignment operator
              
        void transfer(const Tensor<T> &fromTensor,
                      Platform           toPlatform) throw (Exception);
        
        void checkSanity() const;

	//vj: Remove  platform dependency
	//        Platform getPlatform() const;

	//todo :: overload arithmetic operators for tensor class?? -- not really a high priority-- but something that'll be good to have


	~Tensor();
};

//==============================
// Constructors, destructors
//==============================

template<class T>
Tensor<T>::Tensor() {
	buf = NULL;
	dimM = 0;
	dimN = 0;
	dimK = 0;
	dimP = 0;
}

template<class T>
Tensor<T>::~Tensor() {
	delete[] buf;
	buf = NULL;
	dimM = 0;
	dimN = 0;
	dimK = 0;
	dimP = 0;
}

template<class T>
  Tensor<T>::Tensor(size_t M_in, size_t N_in, size_t K_in, size_t P_in) { // , Platform platform_in) {
	//M_in = number of rows in a matrix
	//N_in = number of cols in a matrix
	//K_in = number of rows of matrices

	if (M_in == 0 || N_in == 0 || K_in == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}
	dimM = M_in;
	dimN = N_in;
	dimK = K_in;
	dimP = P_in;
	buf = new (std::nothrow) Matrix<T> [K_in * P_in];
	if (buf == NULL) {
		std::cout << "Error! Cannot allocate memory " << std::endl;
		exit(EXIT_FAILURE);
	} else {
	  for (size_t i = 0; i < K_in * P_in; ++i) {
	    buf[i] = MatrixContainer<T>(M_in, N_in); //, platform_in);
	  }
	}
}

template<class T>
Tensor<T>::Tensor(size_t M_in, size_t N_in, size_t K_in, size_t P_in,
		  matInit_t s) { //, Platform platform_in) {
	//M_in = number of rows
	//N_in = number of cols
	//K_in = number of matrices

	if (M_in == 0 || N_in == 0 || K_in == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}
	dimM = M_in;
	dimN = N_in;
	dimK = K_in;
	dimP = P_in;
	buf = new (std::nothrow) Matrix<T> [K_in * P_in];
	if (buf == NULL) {
		std::cout << "Error! Cannot allocate memory " << std::endl;
		exit(EXIT_FAILURE);
	} else {
		for (size_t i = 0; i < K_in * P_in; ++i) {
                  buf[i] = Matrix<T>(M_in, N_in, s); //, platform_in);
		}
	}
}


//
// template<class T>
//   Platform Tensor<T>::getPlatform() const {
//   return buf[0].getPlatform();
// }


  // This function performs an assignment of fromTensor into this, except
  // - The assignment operator deletes this->buf.
  //   In contrast, transfer() keeps this->buf and checks that its size is same as in fromTensor.
  // - transfer() allows the two tensors to be on different platforms    
  template<class T>
    void Tensor<T>::transfer(const Tensor<T> &fromTensor,
                               Platform           toPlatform) throw (Exception) 
{
  this->     checkSanity();
  fromTensor.checkSanity();                       
         
  if (buf) {
    // Check that this tensor is compatible with what is to come in
    RUDRA_CHECK(dimM == fromTensor.dimM, "transfer() should have: dimM " << dimM << " == " << fromTensor.dimM);
    RUDRA_CHECK(dimN == fromTensor.dimN, "transfer() should have: dimN " << dimN << " == " << fromTensor.dimN);
    RUDRA_CHECK(dimK == fromTensor.dimK, "transfer() should have: dimK " << dimK << " == " << fromTensor.dimK);
    RUDRA_CHECK(dimP == fromTensor.dimP, "transfer() should have: dimP " << dimP << " == " << fromTensor.dimP);
  } else {
    // Initialize this tensor
    dimM = fromTensor.dimM;
    dimN = fromTensor.dimN;
    dimK = fromTensor.dimK;
    dimP = fromTensor.dimP;
    buf  = new Matrix<T>[dimK * dimP];
  }

  // Now transfer the matrices
  for (int k = 0; k < dimK; ++k) {
    for (int p = 0; p < dimP; ++p) {
      buf[k + dimK * p].transfer(fromTensor.buf[k + dimK * p], toPlatform);
    }
  }
 }          

     
  template<class T>
    void Tensor<T>::checkSanity() const
    {
      if (buf == NULL) {
        RUDRA_CHECK(dimM == 0, "dimM = " << dimM);
        RUDRA_CHECK(dimN == 0, "dimN = " << dimN);
        RUDRA_CHECK(dimK == 0, "dimK = " << dimK);
        RUDRA_CHECK(dimP == 0, "dimP = " << dimP);
      } else {
        RUDRA_CHECK(dimM  > 0, "dimM = " << dimM);
        RUDRA_CHECK(dimN  > 0, "dimN = " << dimN);
        RUDRA_CHECK(dimK  > 0, "dimK = " << dimK);
        RUDRA_CHECK(dimP  > 0, "dimP = " << dimP);
      }
    }

//==============================
// Copy constructor
//==============================
template<class T>
Tensor<T>::Tensor(const Tensor<T>& mat) {

	if (mat.dimM == 0 || mat.dimN == 0 || mat.dimK == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D dimension cannot be zero" << std::endl;
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
	this->dimK = mat.dimK;
	this->dimP = mat.dimP;

	this->buf = new (std::nothrow) Matrix<T> [this->dimK * this->dimP];
	if (buf == NULL) {
		std::cout << "Error! Cannot allocate memory " << std::endl;
		exit(EXIT_FAILURE);
	}

	// copy data
	for (size_t i = 0; i < dimK * dimP; ++i) {
		this->buf[i] = mat.buf[i];
	}
}

template<class T>
Tensor<T>::Tensor(const Matrix<T>& mat) {

	if (mat.dimM == 0 || mat.dimN == 0) {
		buf = NULL;
		std::cout << "Error! Matrix dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}

	// allocate new memory
	this->dimM = mat.dimM;
	this->dimN = mat.dimN;
	this->dimK = 1;
	this->dimP = 1;

	this->buf = new (std::nothrow) Matrix<T> [this->dimK * this->dimP];
	if (buf == NULL) {
		std::cout << "Error! Cannot allocate memory " << std::endl;
		exit(EXIT_FAILURE);
	}

	// copy data
	for (size_t i = 0; i < dimK * dimP; ++i) {
          this->buf[i] = mat;
	}
}


//==============================
// Templated copy operator
//==============================

template<class T>
template<class U>
Tensor<T>::Tensor(const Tensor<U>& mat) {

	if (mat.dimM == 0 || mat.dimN == 0 || mat.dimK == 0 || mat.dimP == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D dimension cannot be zero" << std::endl;
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
	this->dimK = mat.dimK;
	this->dimP = mat.dimP;

	this->buf = new (std::nothrow) Matrix<T> [this->dimK * this->dimP];
	if (buf == NULL) {
		std::cout << "Error! Cannot allocate memory " << std::endl;
		exit(EXIT_FAILURE);
	}

	// copy data
	for (size_t i = 0; i < dimK * dimP; ++i) {
		this->buf[i] = mat.buf[i];
	}
}

//==============================
// Assignment operator
//==============================
template<class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& rhs) {

	if (rhs.dimM == 0 || rhs.dimN == 0 || rhs.dimK == 0 || rhs.dimP == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}

	//deallocate existing memory

	if (this->buf != NULL && dimM * dimN * dimK * dimP != 0) {
		this->~Tensor();
	}

	// allocate new memory
	this->dimM = rhs.dimM;
	this->dimN = rhs.dimN;
	this->dimK = rhs.dimK;
	this->dimP = rhs.dimP;

	this->buf = new (std::nothrow) Matrix<T> [this->dimK * this->dimP];
	if (buf == NULL) {
		std::cout << "Error! Tensor3D::Assignment::cannot allocate memory "
				<< std::endl;
		exit(EXIT_FAILURE);
	}

	// copy data
	for (size_t i = 0; i < dimK * dimP; ++i) {
		this->buf[i] = rhs.buf[i];
	}

	return (*this);
}

template<class T>
template<class U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>& rhs) {

	if (rhs.dimM == 0 || rhs.dimN == 0 || rhs.dimK == 0 || rhs.dimP == 0) {
		buf = NULL;
		std::cout << "Error! Tensor3D::dimension cannot be zero" << std::endl;
		exit(EXIT_FAILURE);
	}

	//deallocate existing memory

	if (this->buf != NULL && dimM * dimN * dimK * dimP != 0) {
		this->~Tensor();
	}

	// allocate new memory
	this->dimM = rhs.dimM;
	this->dimN = rhs.dimN;
	this->dimK = rhs.dimK;
	this->dimP = rhs.dimP;

	this->buf = new (std::nothrow) Matrix<T> [this->dimK * this->dimP];
	if (buf == NULL) {
		std::cout
				<< "Error! Tensor3D::TemplatedAssignment::Cannot allocate memory "
				<< std::endl;
		exit(EXIT_FAILURE);
	}

	// copy data
	for (size_t i = 0; i < dimK * dimP; ++i) {
		this->buf[i] = rhs.buf[i];
	}

	return (*this);
}

} /* namespace rudra */
#endif /* RUDRA_MATH_TENSOR_H_ */
