#!/bin/bash

rm -rf hmdb51_org/ hmdb51_org_stips/ non_target_videos/ target_videos/ cluster_centers/
wget -c http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget -c http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org_stips.rar

# Unzipping dataset
unrar x hmdb51_org.rar hmdb51_org/
unrar x -o+ hmdb51_org_stips.rar hmdb51_org_stips/

mkdir target_videos

unrar x hmdb51_org/cartwheel.rar target_videos/
unrar x hmdb51_org/drink.rar target_videos/
unrar x hmdb51_org/ride_bike.rar target_videos/
unrar x hmdb51_org/sword.rar target_videos/
unrar x hmdb51_org/sword_exercise.rar target_videos/
unrar x hmdb51_org/wave.rar target_videos/

unrar x hmdb51_org_stips/cartwheel.rar hmdb51_org_stips/
unrar x hmdb51_org_stips/drink.rar hmdb51_org_stips/
unrar x hmdb51_org_stips/ride_bike.rar hmdb51_org_stips/
unrar x hmdb51_org_stips/sword.rar hmdb51_org_stips/
unrar x hmdb51_org_stips/sword_exercise.rar hmdb51_org_stips/
unrar x hmdb51_org_stips/wave.rar hmdb51_org_stips/

mkdir non_target_videos
mkdir cluster_centers

# Get a list of all the files in the hmdb51_org directory
find hmdb51_org -name "*.rar" -print0 | while IFS= read -r -d $'\0' file; do \
  filename=$(basename "$file" .rar); \
  if [[ "$filename" = "cartwheel" && "$filename" = "drink" && "$filename" = "ride_bike" && "$filename" = "sword" && "$filename" = "sword_exercise" && "$filename" = "wave" ]]; then \
    unrar e "$file" non_target_videos/; \
  fi; \
done