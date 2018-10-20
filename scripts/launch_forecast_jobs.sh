#!/bin/bash
GANFILES=`ls gan_clim_*.sh`
for GANFILE in $GANFILES; do
    echo $GANFILE
    qsub $GANFILE
done
#qsub poly_add_clim.sh
WFILES=`ls gan_weather_n_*.sh`
for WFILE in $WFILES; do
    echo $WFILE
    qsub $WFILE
done
#qsub poly_weather.sh
