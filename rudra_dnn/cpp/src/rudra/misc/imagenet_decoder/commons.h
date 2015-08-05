#ifndef __RUDRA_MISC_COMMONS_H
#define __RUDRA_MISC_COMMONS_H
#include "gflags/gflags.h"
//#include <glog/logging.h> // p775 doesnt have it
//#include <boost/scoped_ptr.hpp> // p775 doesnt have it
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_  really a hack for gflags 

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)
#endif

#define CHECK_EQ 
