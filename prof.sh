#!/bin/bash

nvprof \
 --metrics \
dram_read_bytes,dram_write_bytes,flop_count_sp,inst_integer\
,gld_efficiency,gst_efficiency\
,l2_global_load_bytes,\
,global_hit_rate\
 --events \
global_load,global_store\
,l2_subp0_write_sector_misses,l2_subp1_write_sector_misses,l2_subp0_read_sector_misses,l2_subp1_read_sector_misses\
 ./run 2>&1 | tee logprof.txt
