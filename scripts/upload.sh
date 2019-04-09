#!/bin/sh

#UPLOAD_USER="xxxxx"
#UPLOAD_IP="x.x.x.x"

scp *.py ${UPLOAD_USER}@${UPLOAD_IP}:/home/${UPLOAD_USER}/dendritic_bp
scp scripts/*.sh ${UPLOAD_USER}@${UPLOAD_IP}:/home/${UPLOAD_USER}/dendritic_bp/scripts/
