
#include "../util/Logger.h"
#include "../util/Checking.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <execinfo.h>


namespace rudra {

  void breakpoint()
  {
  }
                 
                 

  static void printBackTrace()
  {
    static const int STACK_SIZE = 100;
    static void *stack[STACK_SIZE];
    
    int size = backtrace(stack, STACK_SIZE);
    char **messages = backtrace_symbols(stack, STACK_SIZE);
    for (int i = 0; i < size; ++i) {
      
      /* find first occurence of '(' or ' ' in message[i] and assume
       * everything before that is the file name. (Don't go beyond 0 though
       * (string terminator)*/
      int p = 0;
      while(messages[i][p] != '(' && messages[i][p] != ' ' && messages[i][p] != 0) {
        ++p;
      }

      char syscom[256];
      sprintf(syscom,"addr2line %p -e %.*s", stack[i], p, messages[i]);
      system(syscom);
    }
    free(messages);
  }
  
  void Checking::reportError(Checking::ERROR_KIND  errorKind,
                             const std::string    &msg,
                             const char           *file,
                             int                   line) throw (Exception)
  {
    breakpoint();
    
    // Add line number and error info to the msg
    std::ostringstream msgStream;
    if (errorKind != USER_ERROR) {
      // Display only one directory of the whole path for file
      const char *s0;  // new beginning of file name
      const char *s1;  // first  /
      const char *s2;  // second /
      for (s0 = file; s0; s0 = s1+1) {
        s1 = strchr(s0+1, '/'); if (!s1) break;
        s2 = strchr(s1+1, '/'); if (!s2) break;
      }
      msgStream << s0 << ":" << line;
    }
    msgStream << " -- " << msg;
    
    if (errorKind != USER_ERROR) {
      // Let user errors be caught and handled in a user friendly manner
      printf("%s\n", msgStream.str().c_str());
      printBackTrace();
    }
    throw  Exception(msgStream.str());
  }

               
} /* namespace rudra */
