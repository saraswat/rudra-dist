/*
 * Platform.h
 *
 *  Created on: May 28, 2015
 *      Author: danbrand
 *
 *  Definitions of platforms
 *
 */

#ifndef RUDRA_UTIL_PLATFORM_H_
#define RUDRA_UTIL_PLATFORM_H_

#include "../util/Exception.h"
 
 namespace rudra {

  typedef enum {
    GPU0 = 0,
    GPU1,
    GPU2,
    GPU3,
    GPU4,
    GPU5,
    GPU6,
    GPU7,
    GPU8,
    GPU9,
    GPU10,
    GPU11,
    GPU12,
    GPU13,
    GPU14,
    GPU15,
    CPU,
    PLATFORM_NUM
  } Platform;


  const char *platformToStr(Platform platform);
  Platform    strToPlatform(std::string s);
  int         platformToDeviceNumber(Platform platform);
  bool        platformIsGpu(Platform platform);
  Platform    deviceNumberToPlatform(int d);
  void        platformCheck(Platform platform) throw (Exception);

} /* namespace rudra */
#endif 
