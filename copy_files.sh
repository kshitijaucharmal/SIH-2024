#!/bin/zsh

# Path to the file you want to share
FILE_PATH="/home/anish/SIH/SIH-2024/multinode.py"

# Destination path on the remote machines
DEST_PATH="/home/pict/SIH1733/sih"

# SSH username
USER="pict"

MACHINES=("10.10.10.112" "10.10.10.116" "10.10.10.107" "10.10.15.94")

# Loop through the machines and use scp to copy the file
for MACHINE in ${MACHINES[@]}
do
    echo "Copying to $MACHINE..."
    
    # Securely copy the file using SCP
    scp "$FILE_PATH" "$USER@$MACHINE:$DEST_PATH"
    
    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "File successfully copied to $MACHINE."
    else
        echo "Failed to copy file to $MACHINE."
    fi
done
