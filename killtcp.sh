#! /bin/bash
kill -9 $(lsof -t -i:9999)
# for i in {8000..8009};do
# kill -9 $(lsof -t -i:$i)
# done

for port in {7999..8009}; do
    # Find the process ID (PID) using the specified port
    pid=$(lsof -t -i:$port)

    # Check if a process is using the port
    if [ -n "$pid" ]; then
        echo "Killing process with PID $pid using port $port"
        kill -9 $pid
    else
        echo "No process found using port $port"
    fi
done

for port in {9000..9002}; do
    # Find the process ID (PID) using the specified port
    pid=$(lsof -t -i:$port)

    # Check if a process is using the port
    if [ -n "$pid" ]; then
        echo "Killing process with PID $pid using port $port"
        kill -9 $pid
    else
        echo "No process found using port $port"
    fi
done