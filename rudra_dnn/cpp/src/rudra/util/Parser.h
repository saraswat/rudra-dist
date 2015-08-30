/*
 * parser.h
 *
 *  Created on: Feb 25, 2015
 *      Author: suyog
 *
 *      This class reads the configuration file and stores the key value pairs in its member object
 */

#ifndef RUDRA_UTIL_PARSER_H_
#define RUDRA_UTIL_PARSER_H_

#include "rudra/util/Logger.h"
#include "rudra/util/Checking.h"
#include <map>
#include <vector>
#include <string>
#include <sstream>

typedef std::map<std::string, std::string> content;
namespace convert{

	template <class T>
	static std::string T_to_string( const T& val){
		std::ostringstream ostr;
		ostr << val;
		return ostr.str();
	}

	template <class T>
	static T string_to_T( const std::string& val){
		std::istringstream istr(val);
		T retval;
		RUDRA_CHECK(static_cast<bool>(istr >> retval), "Convert::string_to_T()::bad input " << val);

		return retval;
	}

        // Parse a comma-separated-value string into a vector of values
	template <class T>
          std::vector<T> csv_to_vector( const std::string& str)
          {
            std::vector<T> answer;
            std::stringstream ss(str);
            
            T v;
            while (ss >> v) {
              answer.push_back(v);
              if (ss.peek() == ',') ss.ignore();
            }
            if (!ss.eof()) {
              rudra::Logger::logFatal("Convert::csv_to_vector()::bad input " + str);
            }
            
            return answer;
          }



};

namespace rudra {

	class Parser {
	public:

		static int lineno;
		std::vector<content> _params;		// create a vector of map objects. Each map object stores the list of key-value pairs for a particular layer.
		std::string 		 _fname;		// file name
		std::fstream		 _fhandle;		// file handle
		int 				 _nblocks;		// number of blocks in the file

		Parser();
		Parser(std::string fname);	//	construct Parser by reading file

		void parseFile();
		void parseBlock();
		void parseLine(std::string l);

	};
	//int Parser::lineno = 0;
	void		removeComments  (std::string&);		//removes comments from the line
	std::string extractKey		(std::string);		// extracts key
	std::string extractValue	(std::string);		// extracts value
	bool		isBlankLine		(std::string);
	bool		isCommentLine	(std::string);
	bool 		isValidLine		(std::string);
	void 		writeFile(std::string path, std::string file, std::vector<content> params);

} /* namespace rudra */

#endif /* RUDRA_UTIL_PARSER_H_ */
