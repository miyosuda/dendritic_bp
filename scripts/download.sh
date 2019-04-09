#!/bin/sh

#UPLOAD_USER="xxxxx"
#UPLOAD_IP="x.x.x.x"

scp ${UPLOAD_USER}@${UPLOAD_IP}:/home/${UPLOAD_USER}/dendritic_bp/saved/*.npz saved/

