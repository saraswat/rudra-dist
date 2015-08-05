
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <sstream>
#include <iostream>
#include "Descriptor.h"
#include "../util/Checking.h"

namespace rudra {



  char *I2::asString() const
  {
    char *answer = (char *) malloc(28);
    sprintf(answer, "[%d,%d]", a, b);
    return answer;
  }

                  

                   
  // Return the index in pos[] containing the smallest number > limit
  static int findMin(const int *pos, 
                     int        limit,
                     int        len)
  {
    int answer;
    int min = INT_MAX;
      
    for (int i = 0; i < len; ++i) {
      if (pos[i] <= limit) continue;
      if (pos[i] <  min) {
        min = pos[i];
        answer = i;
      }
    }
    RUDRA_CHECK(min < INT_MAX, RUDRA_VAR(limit) << RUDRA_VAR(pos[0]) << RUDRA_VAR(pos[1]) << RUDRA_VAR(pos[2]) << RUDRA_VAR(pos[3]));
    return answer;
  }

  // bounds is an array of dimension bounds
  // pos is an array showing the position of each dimension if they were ordered starting from fastest changing
  // Into sizes[] assigns the number of elements for each dimension assuming tight packing
  // Returns total number of elements.
  static I1 calcSizes(const I1  *bounds,
                      const int *pos,
                      I1        *sizes,
                      int        len)   // number of dimensions bounds and pos
  {
    int i;            // index in pos[] containing the smallest number
    int p = -1;       // p = pos[i], initialy smaller than any allowed value of pos
    I1  elements = 1; // All elements represented by bound[] before len
      
    for (int e = 0; e < len; ++e) {
      i         = findMin(pos, p, len);        // next fastest changing dimension 
      p         = pos[i];                      // value used to indicate how fast changing                    
      sizes[i]  = elements;
      elements *= bounds[i];                   // prepare for next dimension e
    }
    return elements;
  }
                   
 
  DataDescriptor::DataDescriptor()
  {
    sizeD.a =  numD.a = -123;
    sizeD.b =  numD.b = -123;
    sizeC   =  numC   = -123;
    sizeN   =  numN   = -123;
    total   = -123;
  }

  DataDescriptor::DataDescriptor(I2 boundsD, int posDa, int posDb,
                                 I1 boundsC, int posC,
                                 I1 boundsN, int posN)
  {
    I1  bounds[4] = {boundsD.a, boundsD.b, boundsC, boundsN};
    int pos[4]    = {posDa, posDb, posC, posN};
    I1  sizes[4];
    total = calcSizes(bounds, pos, sizes, 4);
        
    
    sizeD.a = sizes[0]; numD.a = boundsD.a;
    sizeD.b = sizes[1]; numD.b = boundsD.b;
    sizeC   = sizes[2]; numC   = boundsC;
    sizeN   = sizes[3]; numN   = boundsN;
  }
               
  DataDescriptor::DataDescriptor(I1 boundsDa, int posDa, 
                                 I1 boundsDb, int posDb,
                                 I1 boundsC,  int posC,
                                 I1 boundsN,  int posN)
  {
    I1  bounds[4] = {boundsDa, boundsDb, boundsC, boundsN};
    int pos[4]    = {posDa, posDb, posC, posN};
    I1  sizes[4];
    total = calcSizes(bounds, pos, sizes, 4);
    
    sizeD.a = sizes[0]; numD.a = boundsDa;
    sizeD.b = sizes[1]; numD.b = boundsDb;
    sizeC   = sizes[2]; numC   = boundsC;
    sizeN   = sizes[3]; numN   = boundsN;
  }
                 
                 
  I2 DataDescriptor::getD()     const {return numD;}
  I1 DataDescriptor::getC()     const {return numC;}
  I1 DataDescriptor::getN()     const {return numN;}
  I1 DataDescriptor::getTotal() const {return total;}


  std::string DataDescriptor::boundsAsString()
  {
    std::ostringstream s;
    
    s << "{" << numD.asString() << "," << numC << "," << numN << "}";
    return s.str();
  }
    
  std::string DataDescriptor::layoutAsString()
  {
    std::ostringstream s;
    
    s << "{" << sizeD.asString() << "," << sizeC << "," << sizeN << "}";
    return s.str();
  }
    


  LoweredDataDescriptor::LoweredDataDescriptor()
  {
    sizeQ.a = numQ.a = -123;
    sizeQ.b = numQ.b = -123;
    sizeP.a = numP.a = -123;
    sizeP.b = numP.b = -123;
    sizeC   = numC   = -123;
    sizeN   = numN   = -123;
  }

  LoweredDataDescriptor::LoweredDataDescriptor(I2 boundsQ, int posQa, int posQb,
                                               I2 boundsP, int posPa, int posPb,
                                               I1 boundsC, int posC,
                                               I1 boundsN, int posN)
  {
    I1  bounds[6] = {boundsQ.a, boundsQ.b, boundsP.a, boundsP.b, boundsC, boundsN};
    int pos[6]    = {posQa, posQb, posPa, posPb, posC, posN};
    I1  sizes[6];
    total = calcSizes(bounds, pos, sizes, 6);
        
    sizeQ.a = sizes[0]; numQ.a = boundsQ.a;
    sizeQ.b = sizes[1]; numQ.b = boundsQ.b;
    sizeP.a = sizes[2]; numP.a = boundsP.a;
    sizeP.b = sizes[3]; numP.b = boundsP.b;
    sizeC   = sizes[4]; numC   = boundsC;
    sizeN   = sizes[5]; numN   = boundsN;
  }            

  LoweredDataDescriptor::LoweredDataDescriptor(I1 boundsQa, int posQa, 
                                               I1 boundsQb, int posQb,
                                               I1 boundsPa, int posPa, 
                                               I1 boundsPb, int posPb,
                                               I1 boundsC,  int posC,
                                               I1 boundsN,  int posN)
  {
    I1  bounds[6] = {boundsQa, boundsQb, boundsPa, boundsPb, boundsC, boundsN};
    int pos[6]    = {posQa, posQb, posPa, posPb, posC, posN};
    I1  sizes[6];
    total = calcSizes(bounds, pos, sizes, 6);
        
    sizeQ.a = sizes[0]; numQ.a = boundsQa;
    sizeQ.b = sizes[1]; numQ.b = boundsQb;
    sizeP.a = sizes[2]; numP.a = boundsPa;
    sizeP.b = sizes[3]; numP.b = boundsPb;
    sizeC   = sizes[4]; numC   = boundsC;
    sizeN   = sizes[5]; numN   = boundsN;
  }                       
     
  I2 LoweredDataDescriptor::getQ()     const {return numQ;}
  I2 LoweredDataDescriptor::getP()     const {return numP;}
  I1 LoweredDataDescriptor::getC()     const {return numC;}
  I1 LoweredDataDescriptor::getN()     const {return numN;}
  I1 LoweredDataDescriptor::getTotal() const {return total;}
}
