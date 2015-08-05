To decode lmdb file to Rudra binary file format 
(1) read in the lmdb table, iterate over all the records, convert each record
(a protobuf datum ) to an image. Model after compute_image_mean.cpp
