
#include "../util/Logger.h"
#include "../util/Tracing.h"
#include "../util/Checking.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>


namespace rudra {

  // Variables representing available tracing categories
  // Initially by default no tracing
  int Tracing::timing  = 0;  
  int Tracing::testing = 0;

  // Sets tracing level for category name
  void Tracing::set(const std::string &name,
                    int                val) throw(Exception)
  {
    if      (name == "timing")  timing  = val;
    else if (name == "testing") testing = val;
    
    else RUDRA_CHECK_USER(false, "There is no tracing variable " << name.c_str() << 
                          ". Available variables are 'timing', 'testing'");

  }

               
} /* namespace rudra */
