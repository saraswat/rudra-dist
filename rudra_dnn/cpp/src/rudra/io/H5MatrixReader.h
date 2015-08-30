/*
 * ReadMatrix.h
 */

// vj: removed dependency on platform, i.e. assume platform is CPU.

#ifndef RUDRA_IO_READ_MATRIX_H_
#define RUDRA_IO_READ_MATRIX_H_

#include "rudra/math/MatrixContainer.h"
#include "rudra/math/Tensor.h"
#include "rudra/math/matFunc.h"
#include "rudra/platform/Memory.h"
#include <H5Cpp.h>

namespace rudra {

template<class T>
void readH5(MatrixContainer<T>& mat, H5::Group &group, std::string dataSetName) {
	H5::DataSet dataSet(group.openDataSet(dataSetName));
	H5::DataSpace dataSpace = dataSet.getSpace();

//	if (mat.getPlatform() == CPU) {
		dataSet.read(mat.buf, H5::PredType::NATIVE_FLOAT, dataSpace);
//	} else {
//		size_t numElements = mat.dimM * mat.dimN;
//		float *temp = (float *) malloc(numElements * sizeof(float));
//		dataSet.read(temp, H5::PredType::NATIVE_FLOAT, dataSpace);
//		memoryCopy(mat.buf, mat.getPlatform(), temp, CPU, numElements);
//		free(temp);
//	}
}

// Read the given dataSetName from the given group.
// The result may need to be transposed if memory representation and
// disk respresentation are transposes of each other.
template<class T>
void readH5(Tensor<T>& t, H5::Group &group, std::string dataSetName, bool transpose) {
	H5::DataSet dataSet(group.openDataSet(dataSetName));
	H5::DataSpace dataSpace = dataSet.getSpace();
	hsize_t dims[2];
	dataSpace.getSimpleExtentDims(dims, NULL);

	// read into temporary matrix
	MatrixContainer<float> temp(dims[0], dims[1], _ZEROS, CPU);
	dataSet.read(temp.buf, H5::PredType::NATIVE_FLOAT, dataSpace);

	if (transpose)
		temp.transposeInPlace();

//	if (t.getPlatform() == CPU) {
		// convert Matrix to tensor
		filterMatToTensor(temp, t);
//	} else {
//		t.buf[0].transfer(temp, t.getPlatform());
//	}
}

}
#endif /* RUDRA_IO_READ_MATRIX_H_ */
