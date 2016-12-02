#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: `basename $0` corpus_pos corpus_neg [n_sents_val] [n_sents_test]"
    echo "Shuffles, splits and labels the given corpora"
    echo "If n_sents_val or n_sents_test are given, it will take that number of sentences for val-test (with balanced classes)"
    echo "example:`basename $0` rt-polarity.pos rt-polarity.neg 500 1000"
    echo -e "will create 4 files: \n \t training.sn: Training sentences \n\t training.class: Training classes \n\t  \n \t val.sn: Validation sentences (500) \n\t val.class: Validation classes (500) \n\t  \n \t test.sn: Test sentences (1000) \n\t test.class: Test classes (1000)\n" 
    exit 1
fi

corpus_pos=$1
corpus_neg=$2

if [ $# -ge 3 ]; then
    ndev=$3
fi

if [ $# -ge 4 ]; then
    ntest=$4
fi

destdir=`dirname $corpus_pos`


# Shuffle and label corpora
shuf $corpus_pos |awk '{print $0"\t 1"}'> /tmp/pos
shuf $corpus_neg |awk '{print $0"\t 0"}'> /tmp/neg


if [ $# -ge 3 ]; then
    head -n $((ndev/2))  /tmp/pos > /tmp/dev
    head -n $((ndev/2))  /tmp/neg >> /tmp/dev

    tail -n +$((ndev/2)) /tmp/pos  > /tmp/pos2
    tail -n +$((ndev/2)) /tmp/neg  > /tmp/neg2

else
    cat  /tmp/pos > /tmp/pos2
    rm  /tmp/pos
    cat  /tmp/neg >  /tmp/neg2
    rm  /tmp/neg
fi



if [ $# -ge 4 ]; then
    head -n $((ntest/2))  /tmp/pos2 > /tmp/test
    head -n $((ntest/2))  /tmp/neg2 >> /tmp/test

    tail -n +$((ntest/2)) /tmp/pos2  > /tmp/train
    tail -n +$((ntest/2)) /tmp/neg2  >> /tmp/train

else
    cat  /tmp/pos2 >  /tmp/train
    rm  /tmp/pos2
    cat  /tmp/neg2 >> /tmp/train
    rm  /tmp/neg2
fi

# Shuffle corpora
shuf /tmp/train > /tmp/train_shuf

if [ $# -ge 3 ]; then
    shuf /tmp/dev > /tmp/dev_shuf
fi

if [ $# -ge 4 ]; then
    shuf /tmp/test > /tmp/test_shuf
fi

# Separate class and sentences files
cat /tmp/train_shuf | awk 'BEGIN{FS="\t"}{print $1}' >  $destdir/training.sn
cat /tmp/train_shuf | awk 'BEGIN{FS="\t"}{print $2}' >  $destdir/training.class

if [ $# -ge 3 ]; then
    cat /tmp/dev_shuf | awk 'BEGIN{FS="\t"}{print $1}' >  $destdir/val.sn
    cat /tmp/dev_shuf | awk 'BEGIN{FS="\t"}{print $2}' >  $destdir/val.class
fi

if [ $# -ge 4 ]; then
    cat /tmp/test_shuf | awk 'BEGIN{FS="\t"}{print $1}' >  $destdir/test.sn
    cat /tmp/test_shuf | awk 'BEGIN{FS="\t"}{print $2}' >  $destdir/test.class
fi

# Remove temporal data
rm -f /tmp/*.neg* /tmp/*.pos* /tmp/test* /tmp/train*
