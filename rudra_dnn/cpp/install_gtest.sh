#!/bin/bash

wget -N https://googletest.googlecode.com/files/gtest-1.7.0.zip
unzip gtest-1.7.0.zip
mv gtest-1.7.0 gtest
cd gtest

./configure
make

