#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /az_show.sh
# \brief:
# Author: raphael hao

# view containers in the repository:
az acr repository list --name gcrmembers -o tsv

# get details about a repository:
az acr repository show -n gcrmembers --repository "v-weihaocui/brt"

# get details about a container image version:
az acr repository show -n gcrmembers --image "v-weihaocui/brt:main"

# delete the whole repository:
ac acr repository delete -n gcrmembers --repository "v-weihaocui/brt"

# delete a image:
az acr repository delete -n gcrmembers --image "v-weihaocui/brt:main"
