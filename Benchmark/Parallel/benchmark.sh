#!/bin/bash

cmake .
for glob_gs_exp in `seq 5 9`
do
    for i in `seq 0 6`
    do
        x_num_proc=$((2**($i/3)))
        y_num_proc=$((2**(($i+1)/3)))
        z_num_proc=$((2**(($i+2)/3)))
        echo "running with x_num_proc=${x_num_proc}; y_num_proc=${y_num_proc}; z_num_proc=${z_num_proc}"
        sed -e "s/glob_gs_ph/$((2**$glob_gs_exp))/g" -e "s/x_num_proc_ph/$x_num_proc/g" -e "s/y_num_proc_ph/$y_num_proc/g" -e "s/z_num_proc_ph/$z_num_proc/g" settings.h.template > settings.h
        make
        ./fastmarchingParallelopenmp1
    done
done
