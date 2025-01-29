#!/bin/bash

set -e


file_content='#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0611
"""
Module Description:
    TIP102 - %s - %s - %s - %s

Author: Mike Odnis
Date: %s
Version: 1.0

Usage:
    This module [describe how to use this module]

Dependencies:
    - Python 3.11
    - big_o
    - timeit
    - numpy
"""

# Standard library imports
import os
import sys
import logging
from typing import List, Dict, Tuple, Set, Any, Mapping, Union, Callable, Optional
from collections import Counter, defaultdict, deque, OrderedDict
from functools import lru_cache, cache, wraps, partial
from pathlib import Path
from datetime import datetime
import json
import time
import random
import string

# Third-party imports
import timeit
import big_o
import numpy as np

# Local imports
from utils.data_generators import DataGenerators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEBUG = True
VERSION = "1.0.0"'

log() {
    local level=$1
    shift
    local message=$@
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
}

log_info() {
    log "INFO" "$@"
}

log_error() {
    log "ERROR" "$@"
}

handle_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "An error occurred with exit code $exit_code"
        exit $exit_code
    fi
}

trap handle_error ERR

log_info "Starting directory creation script"

time_start=$(date +%s)
for i in {1..10}
do
    log_info "Processing unit $i"
    for j in {1..2}
    do
        log_info "Creating session $j directories for unit $i"
        mkdir -p "unit_$i/session_$j/standard" || log_error "Failed to create standard directory for unit $i session $j"
        mkdir -p "unit_$i/session_$j/advanced" || log_error "Failed to create advanced directory for unit $i session $j"
        
        for k in {1..2}
        do
            current_date=$(date +%Y-%m-%d)
            if [ $(($k % 2)) -eq 0 ]; then
                log_info "Creating 'two.py' files in unit $i session $j"
                printf "$file_content" "Unit $i" "Session $j" "Standard" "two.py" "$current_date" > "unit_$i/session_$j/standard/two.py" || log_error "Failed to create standard/two.py"
                printf "$file_content" "Unit $i" "Session $j" "Advanced" "two.py" "$current_date" > "unit_$i/session_$j/advanced/two.py" || log_error "Failed to create advanced/two.py"
            else
                log_info "Creating 'one.py' files in unit $i session $j"
                printf "$file_content" "Unit $i" "Session $j" "Standard" "one.py" "$current_date" > "unit_$i/session_$j/standard/one.py" || log_error "Failed to create standard/one.py"
                printf "$file_content" "Unit $i" "Session $j" "Advanced" "one.py" "$current_date" > "unit_$i/session_$j/advanced/one.py" || log_error "Failed to create advanced/one.py"
            fi
        done
    done
done

time_end=$(date +%s%N | cut -c 1-13 | bc)
time_elapsed=$((time_end - time_start))
log_info "Script completed successfully in $time_elapsed milliseconds"
