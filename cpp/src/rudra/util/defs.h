/*
 * defs.h
 *
 * Licensed Materials - Property of IBM
 *
 * Rudra Distributed Learning Platform
 *
 *  Copyright IBM Corp. 2016 All Rights Reserved
 */

#ifndef DEFS_H_
#define DEFS_H_

#include <cstddef>

/* misc typedefs */

namespace rudra {

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int   uint32_t;
typedef unsigned long long  uint64;

typedef signed short int16;
typedef signed int   int32;
typedef long long int64;

#define _RNDMAX 65535
#define _eps 2.224e-16 // as defined in matlab.
#define RUDRA_eps 1e-8


#define  RUDRA_HEADER "\n ################################################################################################### \
	 \n #                                                                                                 # \
     \n #                                                    dddddddd                                     # \
     \n #  RRRRRRRRRRRRRRRRR                                 d::::::d                                     # \
	 \n #  R::::::::::::::::R                                d::::::d                                     # \
	 \n #  R::::::RRRRRR:::::R                               d::::::d                                     # \
	 \n #  RR:::::R     R:::::R                              d:::::d                                      # \
	 \n #    R::::R     R:::::Ruuuuuu    uuuuuu      ddddddddd:::::drrrrr   rrrrrrrrr   aaaaaaaaaaaaa     # \
	 \n #    R::::R     R:::::Ru::::u    u::::u    dd::::::::::::::dr::::rrr:::::::::r  a::::::::::::a    # \
	 \n #    R::::RRRRRR:::::R u::::u    u::::u   d::::::::::::::::dr:::::::::::::::::r aaaaaaaaa:::::a   # \
	 \n #    R:::::::::::::RR  u::::u    u::::u  d:::::::ddddd:::::drr::::::rrrrr::::::r         a::::a   # \
	 \n #    R::::RRRRRR:::::R u::::u    u::::u  d::::::d    d:::::d r:::::r     r:::::r  aaaaaaa:::::a   # \
	 \n #    R::::R     R:::::Ru::::u    u::::u  d:::::d     d:::::d r:::::r     rrrrrrraa::::::::::::a   #	\
	 \n #    R::::R     R:::::Ru::::u    u::::u  d:::::d     d:::::d r:::::r           a::::aaaa::::::a   # \
	 \n #    R::::R     R:::::Ru:::::uuuu:::::u  d:::::d     d:::::d r:::::r          a::::a    a:::::a   # \
	 \n #  RR:::::R     R:::::Ru:::::::::::::::uud::::::ddddd::::::ddr:::::r          a::::a    a:::::a   # \
	 \n #  R::::::R     R:::::R u:::::::::::::::u d:::::::::::::::::dr:::::r          a:::::aaaa::::::a   # \
	 \n #  R::::::R     R:::::R  uu::::::::uu:::u  d:::::::::ddd::::dr:::::r           a::::::::::aa:::a  # \
	 \n #  RRRRRRRR     RRRRRRR    uuuuuuuu  uuuu   ddddddddd   dddddrrrrrrr            aaaaaaaaaa  aaaa  # \
	 \n #                                                                                                 # \
     \n ###################################################################################################"

#define RUDRA_LINEBREAK "#================================================================================================== "
#define RUDRA_HALF_DOTTED_LINE  "#---------------------------------------- "
#define RUDRA_DEFAULT_STRING "_RUDRA_"
#define RUDRA_DEFAULT_INT    (-9090)       // strange, but recognizable
#define RUDRA_DEFAULT_FLOAT  (-9090.9090)  // strange, but recognizable
/* set up global rounding mode:
 * _ROUND_DOWN: Round down
 * _ROUND_UP  : Round up
 * _ROUND_NRST: Round to the nearest
 * _ROUND_RAND: Stochastic rounding
 *
*/

#define _ROUND_RAND true// rounding mode -- 4 options: _ROUND_DOWN, _ROUND_UP, _ROUND_NRST, _ROUND_RAND


enum actFunc_t {
	_SIGMOID, _RELU,		 // rectified linear units
	_IDENTITY,
	_TANH,
	_LWTA,		 // local winner take all
	_SOFTMAX,
};

enum errFunc_t {
	_CE,	// cross-entropy
	_MSE	// mean square error
};

enum poolFunc_t {
	_MAX,			// max pooling
	_AVG,			// average pooling
	_STOCHASTIC,	// todo:stochastic pooling : refer Matt Zeigler's paper
};

} // namespace

#endif /* DEFS_H_ */
