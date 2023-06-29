#!bin/bash

python server.py --ip 127.0.0.1 --port 3001 --world_size 3 --round 3 --ethernet eno1 &

python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 1 --ethernet eno1 &

python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 2 --ethernet eno1  &

wait
