/*
 * ImageUtil.h
 *
 *  Created on: Jul 23, 2015
 *      Author: suyog
 *
 *      Contains the utilities for basic image processing:
 *      1. Mean subtraction
 *      2. Cropping
 *      3. Translation and reflection
 *
 */

#ifndef RUDRA_IMAGEUTIL_H_
#define RUDRA_IMAGEUTIL_H_
#include <rudra/math/Matrix.h>
#include <rudra/util/Checking.h>

namespace rudra {

/* declarations */
template<class T> Matrix<T> subtractMean 	(const Matrix<T> inp, Matrix<T> meanFile);
template<class T> Matrix<T> crop			(const Matrix<T> inp, int dimH, int dimW, int dimC, int croppedHeight, int croppedWidth, std::string pos);
template<class T> Matrix<T> mirror			(const Matrix<T> inp, int dimH, int dimW, int dimC, float probability);


/* in-place versions*/
template<class T> void 		subtractMeanInPlace		(Matrix<T>& inp, Matrix<T> meanFile);
template<class T> void 		cropInPlace				(Matrix<T>& inp, int dimH, int dimW, int dimC, int croppedHeight, int croppedWidth, std::string pos);
template<class T> void 		mirrorInPlace			(Matrix<T>& inp, int dimH, int dimW, int dimC, float probability);


//--------------------------------------
// definitions
//--------------------------------------

template<class T>
Matrix<T> subtractMean(const Matrix<T> inp, Matrix<T> meanImg){

	RUDRA_CHECK(inp.dimN == meanImg.dimN,"subtractMean::dimension mismatch");
	Matrix<T> res = inp;
	inp 	  -= meanImg.repMat(inp.dimM,1);
	return res;
}
template<class T>
void subtractMeanInPlace(Matrix<T>& inp, Matrix<T> meanImg){

	RUDRA_CHECK(inp.dimN == meanImg.dimN,"subtractMean::dimension mismatch");
	inp 	  -= meanImg.repMat(inp.dimM,1);
}

template<class T>
Matrix<T> crop(const Matrix<T> inp, int dimH, int dimW, int dimC, int croppedHeight, int croppedWidth, std::string pos){
	//expects inp to be linearized image
	// dimH = number of rows = image height
	// dimW = number of cols = image width
	// dimC = number of channels

	RUDRA_CHECK(inp.dimN == dimH*dimW*dimC,"crop::dimension mismatch ");

	// possible values of pos: topleft, topright, bottomleft, bottomright, center, random
	int startRow, startCol;

	if(pos=="topleft"){
		startRow = 0;
		startCol = 0;

	}else if(pos == "topright"){
		startRow = 0;
		startCol = dimW - croppedWidth;

	}else if(pos == "bottomleft"){
		startRow = dimH - croppedHeight;
		startCol = 0;

	}else if(pos == "bottomright"){
		startRow = dimH - croppedHeight;
		startCol = dimW - croppedWidth;


	}else if(pos == "random"){
		startRow = (rand())%(dimH - croppedHeight);
		startCol = (rand())%(dimW - croppedWidth);
	}
	else {
		startRow = (dimH - croppedHeight)/2;
		startCol = (dimW - croppedWidth)/2;
	}


	Matrix<T> res(inp.dimM,croppedHeight*croppedWidth*dimC,_ZEROS);

	for (int ss = 0; ss < inp.dimM; ++ss){

		for (int mm = 0; mm < dimC; ++mm){
			for (int rr = 0; rr < croppedHeight; ++rr){
				for(int cc = 0; cc < croppedWidth; ++cc){
					res.buf[cc + rr*croppedWidth + mm*croppedHeight*croppedWidth + ss*croppedHeight*croppedWidth*dimC]
							= inp.buf[(cc+startCol) + (rr+startRow)*dimW + mm*dimW*dimH + + ss*dimW*dimH*dimC];
				}
			}
		}
	}
	return res;

}


template<class T>
void cropInPlace(Matrix<T>& inp, int dimH, int dimW, int dimC, int croppedHeight, int croppedWidth, std::string pos){
	//expects inp to be linearized image
	// dimH = number of rows = image height
	// dimW = number of cols = image width
	// dimC = number of channels

	RUDRA_CHECK(inp.dimN == dimH*dimW*dimC,"crop::dimension mismatch ");

	// possible values of pos: topleft, topright, bottomleft, bottomright, center, random
	int startRow, startCol;

	if(pos=="topleft"){
		startRow = 0;
		startCol = 0;

	}else if(pos == "topright"){
		startRow = 0;
		startCol = dimW - croppedWidth;

	}else if(pos == "bottomleft"){
		startRow = dimH - croppedHeight;
		startCol = 0;

	}else if(pos == "bottomright"){
		startRow = dimH - croppedHeight;
		startCol = dimW - croppedWidth;


	}else if(pos == "random"){
		startRow = (rand())%(dimH - croppedHeight);
		startCol = (rand())%(dimW - croppedWidth);
	}
	else {
		startRow = (dimH - croppedHeight)/2;
		startCol = (dimW - croppedWidth)/2;
	}


	Matrix<T> res(inp.dimM,croppedHeight*croppedWidth*dimC,_ZEROS);

	for (int ss = 0; ss < inp.dimM; ++ss){

		for (int mm = 0; mm < dimC; ++mm){
			for (int rr = 0; rr < croppedHeight; ++rr){
				for(int cc = 0; cc < croppedWidth; ++cc){
					res.buf[cc + rr*croppedWidth + mm*croppedHeight*croppedWidth + ss*croppedHeight*croppedWidth*dimC]
							= inp.buf[(cc+startCol) + (rr+startRow)*dimW + mm*dimW*dimH + + ss*dimW*dimH*dimC];
				}
			}
		}
	}
	inp = res;

}


template<class T>
Matrix<T> mirror(const Matrix<T> inp, int dim0, int dim1, int dim2, float probability){

	// dim0 = image height
	// dim1 = image width
	// dim2 = number of maps
	RUDRA_CHECK(inp.dimN == dim0*dim1*dim2,"mirrorInPlace::dimension mismatch ");

	Matrix<T> res(inp.dimM, inp.dimN, _ZEROS);
	if(probability > float(rand()%_RNDMAX)/_RNDMAX){

		for (int rr =0; rr < inp.dimM; ++rr){
			for (int mm = 0; mm < dim2; ++mm){
			// for each channel
				for (int hh = 0; hh < dim0; ++hh){
				// for each row of the image
					for (int kk = 0; kk < dim1; ++kk){
						res(rr, mm*dim0*dim1 + hh*dim1 + kk) = inp(rr, mm*dim0*dim1 + hh*dim1 + dim1-kk-1);

					}
				}
			}

		}
		return res;

	}else{
		return inp;
	}

}

template<class T>
void mirrorInPlace(Matrix<T>& inp, int dim0, int dim1, int dim2, float probability){
	Matrix<T> res(inp.dimM, inp.dimN, _ZEROS);

	// dim0 = image height
	// dim1 = image width
	// dim2 = number of maps
	RUDRA_CHECK(inp.dimN == dim0*dim1*dim2,"mirrorInPlace::dimension mismatch ");

	if(probability > float(rand()%_RNDMAX)/_RNDMAX){

		for (int rr =0; rr < inp.dimM; ++rr){
			for (int mm = 0; mm < dim2; ++mm){
			// for each channel
				for (int hh = 0; hh < dim0; ++hh){
				// for each row of the image
					for (int kk = 0; kk < dim1; ++kk){
						res(rr, mm*dim0*dim1 + hh*dim1 + kk) = inp(rr, mm*dim0*dim1 + hh*dim1 + dim1-kk-1);

					}
				}
			}

		}
		inp = res;

	}
}


} /* namespace rudra */


#endif /* RUDRA_IMAGEUTIL_H_ */
