#include "db.hpp"
#include "db_lmdb.hpp"

#include <string>
#include <iostream>
namespace rudra { namespace db {


  DB* GetDB(const std::string& backend) {
  if (backend == "lmdb") {
      return new LMDB();
  } else {
      //    LOG(FATAL) << "Unknown database backend";
      std::cerr<<"Unknown database backend"<<std::endl;
      //exit(-1);
  }
}

}  // namespace db
}  // namespace rudra
