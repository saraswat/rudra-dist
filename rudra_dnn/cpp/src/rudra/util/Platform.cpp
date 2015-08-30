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
    case CPU:   return "CPU";
    case GPU0:  return "GPU0";
    case GPU1:  return "GPU1";
    case GPU2:  return "GPU2";
    case GPU3:  return "GPU3";
    case GPU4:  return "GPU4";
    case GPU5:  return "GPU5";
    case GPU6:  return "GPU6";
    case GPU7:  return "GPU7";
    case GPU8:  return "GPU8";
    case GPU9:  return "GPU9";
    case GPU10: return "GPU10";
    case GPU11: return "GPU11";
    case GPU12: return "GPU12";
    case GPU13: return "GPU13";
    case GPU14: return "GPU14";
    case GPU15: return "GPU15";
    default: break;
    } 
    return "Invalid Platform";
  }

  Platform  strToPlatform(std::string s)
  {
    if (s == "CPU")  return CPU;
    if (s == "GPU0") return GPU0;
    if (s == "GPU1") return GPU1;
    if (s == "GPU2") return GPU2;
    if (s == "GPU3") return GPU3;
    if (s == "GPU4") return GPU4;
    if (s == "GPU5") return GPU5;
    if (s == "GPU6") return GPU6;
    if (s == "GPU7") return GPU7;
    if (s == "GPU8") return GPU8;
    if (s == "GPU9") return GPU9;
    if (s == "GPU10") return GPU10;
    if (s == "GPU11") return GPU11;
    if (s == "GPU12") return GPU12;
    if (s == "GPU13") return GPU13;
    if (s == "GPU14") return GPU14;
    if (s == "GPU15") return GPU15;
    return PLATFORM_NUM;
  }

  int platformToDeviceNumber(Platform platform) 
  {
    RUDRA_CHECK(platform >= 0,  "Invalid platform " << platformToStr(platform));
    RUDRA_CHECK(platform < CPU, "Invalid platform " << platformToStr(platform));
   
    // The enum is defined so that enumerator values are device IDs 
    return (int) platform;
  }

  Platform deviceNumberToPlatform(int d) 
  {
    if (0 <= d && d < CPU) return (Platform) d;
    return PLATFORM_NUM;
  }


  // Check that platform has a valid values, and
  // it is allowed to be GPU only if RUDRA_CUDA is defined
  void platformCheck(Platform platform) throw (Exception)
  {
    RUDRA_CHECK(platform >= 0,           "Invalid platform " << platformToStr(platform));
    RUDRA_CHECK(platform < PLATFORM_NUM, "Invalid platform " << platformToStr(platform));
    
    if (platformIsGpu(platform)) {// allowed only if RUDRA_CUDA defined
#ifndef RUDRA_CUDA
      RUDRA_CHECK(false, "GPU platform without RUDRA_CUDA defined");
#endif
    } 
  }

  bool platformIsGpu(Platform platform)
  {
    return GPU0 <= platform && platform < CPU;
  }
  

} /* namespace rudra */
