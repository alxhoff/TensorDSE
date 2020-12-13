#!/bin/sh

DURATION=60
INTERFACE="usbmon0"
O_FILE="$HOME/Downloads/capture.cap"

BEGIN=$(date +%s)

tshark -i $INTERFACE -w $O_FILE -a duration:${DURATION}

END=$(date +%s)

DIFF=$((END-BEGIN))

echo "Time Duration: $DIFF"
#echo "TS Begin: $BEGIN"
#echo "TS End: $END"
