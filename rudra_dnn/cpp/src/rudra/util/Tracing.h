

#ifndef RUDRA_UTIL_TRACING_H_
#define RUDRA_UTIL_TRACING_H_

#include "../util/Exception.h"

namespace rudra {

  class Tracing
  {
  public:
  
    static int timing;
    static int testing;

    static void set(const std::string &name,
                    int                val) throw(Exception);
  }; 
}



#define RUDRA_TRACE(name, val) if (Tracing::name >= val)

#endif /* RUDRA_UTIL_TRACING_H_ */
