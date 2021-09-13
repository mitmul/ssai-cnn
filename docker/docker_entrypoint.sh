#!/usr/bin/env bash

ln -s /dev/null /dev/raw1394
groupadd user -g ${GROUP_ID} -o
useradd -u ${USER_ID} -o --create-home --home-dir /home/user -g user user
exec gosu user "${@}"
