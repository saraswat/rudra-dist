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
    CPU = 0,
    GPU,
    PLATFORM_NUM
  } Platform;


  const char *platformToStr(Platform platform);
  Platform    strToPlatform(std::string s);
  void        platformCheck(Platform platform) throw (Exception);

} /* namespace rudra */
#endif 
