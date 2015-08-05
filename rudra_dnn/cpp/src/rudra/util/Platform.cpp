/*
 * Platform.cpp
 *
 *  Created on: May 28, 2015
 *      Author: danbrand
 *
 *      Operations on the Platform
 */


#include "Platform.h"
#include "../util/Checking.h"

namespace rudra {

  const char *platformToStr(Platform platform) 
  {
    switch (platform) {
    case CPU: return "CPU";
    case GPU: return "GPU";
    default: break;
    } 
    return "Invalid Platform";
  }

  Platform  strToPlatform(std::string s)
  {
    if (s == "CPU") return CPU;
    if (s == "GPU") return GPU;
    
    RUDRA_CHECK_USER(false, "Invalid platform " << s);
    return CPU;
  }

  // Check that platform has a valid values, and
  // it is allowed to be GPU only if RUDRA_CUDA is defined
  void platformCheck(Platform platform) throw (Exception)
  {
    switch (platform) {
    case CPU: { // always allowed
      break; 
    }
    case GPU: { // allowed only if RUDRA_CUDA defined
#ifndef RUDRA_CUDA
      RUDRA_CHECK(false, "GPU platform without RUDRA_CUDA defined");
#endif
      break;
    }
      default: RUDRA_CHECK(false, "Invalid platform " << platformToStr(platform));
    } 
  }
  

} /* namespace rudra */
