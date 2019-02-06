#!/usr/bin/env bash

BOT_FOLDER="/home/user/.local/share/Steam/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots"

# remove old bots' folder
rm -rf "$BOT_FOLDER"
# copy up-to-date bots' code to Dota 2 folder
cp -r bot "$BOT_FOLDER"
