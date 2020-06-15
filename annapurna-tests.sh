#!/bin/bash

testFiles="tests/testFiles"
outputsDir="tests/outputs"
expectedOutputsDir="tests/outputs_expected"


function compare {
  for f in $(find $expectedOutputs/ -name *.pdb -or -name *.csv -or -name *.phar -type f); do
    newFile=$outDir/$(basename $f)
    echo "Comparing $f with $newFile..."

    if diff -q $f $newFile; then
      echo -e "\e[92mOK\e[39m"
    else
      echo "\e[41mDifferent\e[49m"
      exit 1
    fi

  done
}


# ------------------------------------------------------------------------- #

method=kNN
echo "*** testing $method method"
outDir="$outputsDir/$method"
expectedOutputs="$expectedOutputsDir/$method"

mkdir -p $outDir
cp $testFiles/* $outDir/
./annapurna.py -r $outDir/1AJU.pdb -l $outDir/ARG.sdf -m kNN_modern -o $outDir/output -s --overwrite --groupby --merge
compare

# ------------------------------------------------------------------------- #

method=kNN_clust
echo "*** testing $method method"
outDir="$outputsDir/$method"
expectedOutputs="$expectedOutputsDir/$method"

mkdir -p $outDir
cp $testFiles/* $outDir/
./annapurna.py -r $outDir/1AJU.pdb -l $outDir/ARG.sdf -m kNN_modern -o $outDir/output -s --overwrite --groupby --merge --cluster_fraction 1.0 --cluster_cutoff 2.0 --clustering_method AD
compare

# ------------------------------------------------------------------------- #

# starting h2o.ai server (will kill it after this test script finishes)
./start_h2o.sh 1> /dev/null 2>&1 &
sleep 2s

# ------------------------------------------------------------------------- #


method=DL
echo "*** testing $method method"
outDir="$outputsDir/$method"
expectedOutputs="$expectedOutputsDir/$method"

mkdir -p $outDir
cp $testFiles/* $outDir/
./annapurna.py -r $outDir/1AJU.pdb -l $outDir/ARG.sdf -m DL_modern -o $outDir/output -s --overwrite --groupby --merge -p 30000
compare

# ------------------------------------------------------------------------- #

## killing the h2o server
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
