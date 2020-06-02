#!/bin/bash

function processfile () {
bzcat $1 | head -n 50000 | jq -c ".text" | bzip2 >  $1.text
bzcat $1 | head -n 50000 | jq -c "[.entities.hashtags[].text]" | bzip2 > $1.hashtags
}

export -f processfile
ls ./data/*.bz2 | parallel -j 40 processfile {}
