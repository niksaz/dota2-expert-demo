#!/usr/bin/env bash

# remove old bots' folder
rm -rf ~/Library/Application\ Support/Steam/steamapps/common/dota\ 2\ beta/game/dota/scripts/vscripts/bots
# copy currect bots' code to DotA2 folder
cp -r bot ~/Library/Application\ Support/Steam/steamapps/common/dota\ 2\ beta/game/dota/scripts/vscripts/bots
