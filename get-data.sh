#!/bin/bash

pushd .
cd data/shakespeare/

if [ -e shakespeare-12th.txt ]
then
    echo Using existing Shakespearian dataset.
else
    echo Dowloading Shakespearian dataset.
    
    mkdir shakespeare
    pushd shakespeare
    curl https://www.gutenberg.org/files/1526/1526.zip > shakespeare-12th.zip
    unzip shakespeare-12th.zip
    popd
fi

popd