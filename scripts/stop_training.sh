#!/bin/sh

ps aux | grep "python3 main.py" | grep -v grep | awk '{ print "kill -9", $2 }' | sh
#ps aux | grep "tensorboard" | grep -v grep | awk '{ print "kill -9", $2 }' | sh
