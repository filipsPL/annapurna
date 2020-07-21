#!/bin/bash

source activate annapurna

port=30000

echo "Starting h2o server at port $port ..."
java -jar external_software/h2o-3.9.1.3501/h2o.jar -port $port -name annapurna_${port}
