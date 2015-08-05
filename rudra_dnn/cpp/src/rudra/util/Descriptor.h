/* 
 * This defined tensor descriptors specifying how a tensor should be accessed.
 * Some methods in the classes here are defined to run on CPU and others on GPU.
 * Since this file can be included from both C++ and CUDA source files,
 * protect cuda code by both RUDRA_CUDA and the nvcc's __CUDACC__
 */

#ifndef RUDRA_CUDA_DESCRIPTOR_H_
#define RUDRA_CUDA_DESCRIPTOR_H_

#include <string>

namespace rudra {

  // Counter of cuda objects: threads, blocks, ...
  // Unfortunately the cuda guys are inconsistent -- 
  // dim3 contains unsigned ints, but cudaDeviceProp contains ints.
  // We will use int.
  typedef int C1;

  // Index for matrix dimension, indices, and for counting desired GPU threads.
  // It is important that it be a signed type, and 
  // that it be at least as big as C1.
  typedef int I1;
  
  // A pair of matrix indices (pair because 2D convolution)
  class I2
  {
  public:
    I1    a;
    I1    b;

#ifdef __CUDACC__
#ifdef RUDRA_CUDA
    __device__ I2 operator *(const I2 &other) const
      {
        I2 answer = {a * other.a, b * other.b};
        return answer;
      }

    __device__ I2 operator +(const I2 &other) const
    {
      I2 answer = {a + other.a, b + other.b};
      return answer;
    }

    __device__ I2 operator -(const I2 &other) const
    {
      I2 answer = {a - other.a, b - other.b};
      return answer;
    }
      
    __device__ bool operator <(const I2 &other) const
    {
      return a < other.a && b < other.b;
    }

    __device__ bool operator >=(I1 c) const
    {
      return a >= c && b >= c;
    }

    __device__ I2 reducedTo(const I2 &other) const
    {
      I2 answer = {a < other.a ? a : other.a,
                   b < other.b ? b : other.b};
      return answer;
    }

    __device__ I2 increasedTo(I1 c) const
    {
      I2 answer = {a < c ? c : a,
                   b < c ? c : b};
      return answer;
    }
    __device__ I1 area() const
    {
      return a * b;
    }
#endif
#endif

    char *asString() const;
  };




  // Describes dimensions and layout of data (and gradients) consisting of
  // 2 dimensional frame     D
  // 1 dimensional color     C
  // 1 dimensional minibatch N

  class DataDescriptor {
                        
  private:
    I2 sizeD; /* floats in one frame */  I2 numD; /* number of frames */
    I1 sizeC; /* floats in one color */  I1 numC; /* number of colors */
    I1 sizeN; /* floats in one sample */ I1 numN; /* number of samples */
                        
    I1 total; /* floats in the whole array */
                   
  public:
    DataDescriptor();
       
    DataDescriptor(I2 boundsD, int posDa, int posDb,
                   I1 boundsC, int posC,
                   I1 boundsN, int posN);
                   
    DataDescriptor(I1 boundsDa, int posDa, 
                   I1 boundsDb, int posDb,
                   I1 boundsC,  int posC,
                   I1 boundsN,  int posN);
                        
    I2 getD() const;
    I1 getC() const;
    I1 getN() const;
    I1 getTotal() const;
                        
    std::string boundsAsString();
    std::string layoutAsString();

#ifdef __CUDACC__
#ifdef RUDRA_CUDA
    
    __device__  I2 get_D() const {return numD;}
  
    __device__ I1 isInBounds(I2 d,
                             I1 c,
                             I1 n)
    {
      return (d.a < numD.a &&
              d.b < numD.b &&
              c   < numC   &&
              n   < numN);
    }

    __device__ I1 linearIndex(I2 d,
                              I1 c,
                              I1 n)
    {
      return (d.a * sizeD.a +
              d.b * sizeD.b +
              c   * sizeC   +
              n   * sizeN);
    }


#endif
#endif

  };

  // Describes dimensions and layout of lowered data (and gradients) consisting of
  // 2 dimensional window    Q
  // 2 dimensional positions P
  // 1 dimensional color     C
  // 1 dimensional minibatch N
                 
  class LoweredDataDescriptor {
                               
  private:
    I2 sizeQ; /* floats in one kernel element */  I2 numQ; /* number of kernels elements in a kernel */
    I2 sizeP; /* floats in one kernel position */ I2 numP; /* number of kernel positions */
    I1 sizeC; /* floats in one color */           I1 numC; /* number of colors */
    I1 sizeN; /* floats in one sample */          I1 numN; /* number of samples */
                        
    I1 total; /* floats in the whole array */
                      
  public:         
    LoweredDataDescriptor();
                               
    LoweredDataDescriptor(I2 boundsQ, int posQa, int posQb,
                          I2 boundsP, int posPa, int posPb,
                          I1 boundsC, int posC,
                          I1 boundsN, int posN);

    LoweredDataDescriptor(I1 boundsQa, int posQa, 
                          I1 boundsQb, int posQb,
                          I1 boundsPa, int posPa, 
                          I1 boundsPb, int posPb,
                          I1 boundsC,  int posC,
                          I1 boundsN,  int posN);
    
    I2 getQ() const;
    I2 getP() const;
    I1 getC() const;
    I1 getN() const;
    I1 getTotal() const;

#ifdef __CUDACC__
#ifdef RUDRA_CUDA
    
    __device__ I1 linearIndex (I2 q,
                               I2 p,
                               I1 c,
                               I1 n)
    {
      return (q.a * sizeQ.a +
              q.b * sizeQ.b +
              p.a * sizeP.a +
              p.b * sizeP.b +
              c   * sizeC   +
              n   * sizeN);
    }
#endif
#endif
  };
}
#endif
