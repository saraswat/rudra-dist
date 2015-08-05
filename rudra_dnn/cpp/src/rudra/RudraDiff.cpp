#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string>
#include "util/Checking.h"
#include "util/Parser.h"

#define MAX_STRING_LEN 100
#define MAX_STRING_PAT "%100s"

// Essentially performs the function of 'diff', but 
// filters tailored to comparing rudra output files:
// - filter 1: "CPU" and "GPU" are conidered the same because we want to compare CPU and GPU runs
// - filter 2: numeric differences within small relative tolerance are ignored

// It returns
//   0 if the files are considered same
//   1 if neither file exists, which is used by invoking script to stop recursion
// > 1 otherwise

namespace rudra {
                 

  static int compare(int argc, const char *argv[])
  {
    RUDRA_CHECK(argc == 3, RUDRA_VAR(argc));

    char str1[MAX_STRING_LEN+1];
    char str2[MAX_STRING_LEN+1];
  
    const char *fileName1 = argv[1];
    const char *fileName2 = argv[2];
    
    RUDRA_CHECK(fileName1, "");
    RUDRA_CHECK(fileName2, "");
  
    FILE *file1 = fopen(fileName1, "r");
    FILE *file2 = fopen(fileName2, "r");
    
    // If both files do not exist that is an indication that we have compared
    // everything already -- return non-0 code
    if (!file1 && !file2) return 1;
  
    RUDRA_CHECK(file1, "Failed to open " << RUDRA_VAR(fileName1));
    RUDRA_CHECK(file2, "Failed to open " << RUDRA_VAR(fileName2));
    
    float maxAbsoluteDiff = 0;
    float maxRelativeDiff = 0;
    
    try {
      for (;;) {
        int rc1 = fscanf(file1, MAX_STRING_PAT, str1);
        int rc2 = fscanf(file2, MAX_STRING_PAT, str2);

        if (rc1 <= 0) {
          if (rc2 >  0) {printf("%s is shorter than %s\n", fileName1, fileName2); return 11;}
        } else {
          if (rc2 <= 0) {printf("%s is longer  than %s\n", fileName1, fileName2); return 12;}
        }
        if (rc1 <= 0 || rc2 <= 0) break;
      
        if (strcmp(str1, str2) != 0) {

          // filter 1:
          if (strcmp(str1, "CPU") == 0 && strcmp(str2, "GPU") == 0) continue;
          if (strcmp(str1, "GPU") == 0 && strcmp(str2, "CPU") == 0) continue;
                    
          // Calculate differences for filter 2:                

          std::vector<float> vv1 = convert::csv_to_vector<float>(str1);
          std::vector<float> vv2 = convert::csv_to_vector<float>(str2);
          if (vv1.size() != vv2.size()) throw Exception("");
                                      
          for (int i = 0; i < vv1.size(); ++i) {
            float v1 = vv1[i];
            float v2 = vv2[i];
            
            float d = fabs(v1 - v2);
            float m = (fabs(v1) > fabs(v2)) ? fabs(v1) : fabs(v2);
            float r = d/m;
            
            if (maxAbsoluteDiff < d) maxAbsoluteDiff = d;
            if (maxRelativeDiff < r) maxRelativeDiff = r;
          }
        }
      }
    } catch (Exception ) {
      RUDRA_CHECK_USER(false, "Invalid numbers '" << str1 << ", " << str2 << " in " << fileName1 << ", " << fileName2);
      return 13;
    }                                      

    // filter 2:
    if (maxAbsoluteDiff > 0.0001) {
      printf("maxAbsoluteDiff = %g, maxRelativeDiff = %g \n", maxAbsoluteDiff, maxRelativeDiff);
      return 15;
    } else {
      return 0;
    }
  }


}
int main(int argc, const char *argv[])
{
  int rc = rudra::compare(argc, argv);
  return rc;
}
