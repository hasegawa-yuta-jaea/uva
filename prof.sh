#!/bin/bash

nvprof \
 --metrics \
dram_read_bytes,dram_write_bytes,flop_count_sp,inst_integer\
,gld_efficiency,gst_efficiency\
,l2_global_load_bytes,\
,global_hit_rate,\
,branch_efficiency\
 --events \
global_load,global_store\
,l2_subp0_total_read_sector_queries,l2_subp1_total_read_sector_queries\
,l2_subp0_total_write_sector_queries,l2_subp1_total_write_sector_queries\
 ./run 2>&1 | tee logprof.txt
