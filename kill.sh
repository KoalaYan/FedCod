#! /bin/bash
process=$2
pid=$(ps -aux | grep "from multiprocessing.resource_tracker" | grep -v grep | awk '{print $2}')
for i in $pid;do
kill -9 $i
echo $i
done
pid2=$(ps -aux | grep "from multiprocessing.spawn import spawn_main" | grep -v grep | awk '{print $2}')
for i in $pid2;do
kill -9 $i
echo $i
done