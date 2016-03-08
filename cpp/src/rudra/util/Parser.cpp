/*
 * Parser.cpp
 *
 * Licensed Materials - Property of IBM
 *
 * Rudra Distributed Learning Platform
 *
 *  Copyright IBM Corp. 2016 All Rights Reserved
 */

#include "rudra/util/Parser.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace rudra {
  int Parser::lineno = 0;
	Parser::Parser() {
		//nothing to do
		_nblocks = 0;
	}

	Parser::Parser(std::string fname){

		_fname 	 = fname;
		_nblocks = 0;

		std::string s = _fname;
		_fhandle.open(s.c_str(),std::ios::in);

		if(!_fhandle){
			rudra::Logger::logFatal("Parser: Could not open file: " + s);
		}

	}


	void Parser::parseFile(){
		Parser::lineno =0;
		rudra::Logger::logInfo("Parser: Parsing file: " + _fname + " ... ");

		std::string line;

		while(!_fhandle.eof()){
			// continue till the end of file
			getline(_fhandle,line);
			Parser::lineno++;
			if(isBlankLine(line) || isCommentLine(line)) continue;
			removeComments(line);


			if(line.find('{') < line.npos){
				parseBlock();
				_nblocks ++;
			}

		}
		std::ostringstream msg;
		msg << "Parser: Parsed a total of " << _nblocks << " block(s)";
		rudra::Logger::logInfo(msg.str());
		_fhandle.close();
	}


	void Parser::parseBlock(){
		std::string line;
		content p;
		_params.push_back(p);
		getline(_fhandle,line); Parser::lineno++;
		while(line.find('}')==line.npos){

			if(isBlankLine(line) || isCommentLine(line)) {
				getline(_fhandle,line);
				Parser::lineno++;
				continue;
			}
			removeComments(line);
			parseLine(line);
			getline(_fhandle,line);
			Parser::lineno++;
		}



	}

	void Parser::parseLine(std::string line){

		if(!isValidLine(line)){
			std::ostringstream msg;
			msg << " invalid line number " << Parser::lineno << "\t" << line;
			rudra::Logger::logFatal(msg.str());
		}

		_params[_nblocks].insert(std::pair<std::string, std::string>(extractKey(line),extractValue(line)));

	}
  void removeComments (std::string& line){
	if(line.find('#')!=line.npos)
		line.erase(line.find('#'));
  }






bool isBlankLine (std::string line){
	return(line.find_first_not_of(' ') == line.npos);
}

bool isCommentLine (std::string line){
	return(line[line.find_first_not_of(' ')] == '#');
}


bool isValidLine (std::string line){
	std::string temp = line;
	
	if(temp.find('=') == line.npos){
		rudra::Logger::logError("Parser: Invalid line, Missing \'=\' in ");
		return false;
		
	}
	
	temp.erase(0,temp.find_first_not_of("\t "));
	if(temp[0]== '='){
		rudra::Logger::logError("Parser: Invalid line, Missing key in ");
		return false;
	}
	
    for (size_t i = temp.find('=') + 1; i < temp.length(); ++i) {
		if(temp[i]!= ' '){
			return true;
		}
	}
	
    rudra::Logger::logError("Parser: Invalid line, Missing value in ");
	return false;
	
}

std::string extractKey(std::string line){
	std::string temp = line.substr(0,line.find('='));
	temp.erase(0,temp.find_first_not_of("\t "));
	temp.erase(temp.find_last_not_of("\t ") + 1);
	return temp;
}

std::string extractValue(std::string line){
	std::string temp = line.substr(line.find('=')+1,line.length()-1);
	temp.erase(0,temp.find_first_not_of("\t "));
	temp.erase(temp.find_last_not_of("\t ") + 1);
	return temp;
}

void writeFile(std::string path, std::string file, std::vector<content> params){
	std::string s = path + file;
	std::fstream fhandle;
	fhandle.open(s.c_str(),std::ios::out);

    for (size_t i = 0; i < params.size(); ++i) {
		fhandle << "\n{" << std::endl;
		content::iterator it = params[i].begin();
		for(content::iterator it = params[i].begin(); it != params[i].end(); it++){
			 fhandle << it->first << "="<<it->second << std::endl;
		}
		fhandle << "}" << std::endl;
	}

}
};
