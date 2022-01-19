#!/bin/bash
date +"Date : %d/%m/%Y Time : %H.%M.%S"

REPO_URL="https://github.com/oscar-defelice/DeepLearning-lectures/"
################################################################################
# Help                                                                         #
################################################################################
Help()
{
   # Display Help
   echo
   echo -e "\033[1mThis script performs the pull of lectures overwriting \033[0m"
   echo -e "\033[1mthe local content. \033[0m"
   echo
   echo -e "\033[1mSyntax\033[0m: ./fullpull [h] [options]"
   echo
#   echo -e "\033[1margument\033[0m:"
#   echo "username  the Twitter username"
   echo -e "\033[1moptions\033[0m:"
   echo "h         print this help."
#   echo "n         the maximum number of tweets to download"
#   echo "o         the output file path and name"
}

while getopts ":h" option;
do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

################################################################################
# Main script                                                                  #
################################################################################


printf "Date: %s\nWe are now pulling overwriting local content from %s\n" "$date" "$REPO_URL"
git stash push --include-untracked
git stash drop
git pull -f
