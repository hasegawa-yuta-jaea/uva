#!/bin/bash

nvprof \
 --metrics \
dram_read_bytes,dram_write_bytes,flop_count_sp,inst_integer\
,gld_efficiency,gst_efficiency,\
,global_hit_rate\
 --events \
generic_load,generic_store,global_load,global_store\
 ./run 2>&1 | tee logprof.txt
