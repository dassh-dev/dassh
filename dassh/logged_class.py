########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2020-10-06
author: matz (inherited from Dr. A. Nelson)
Log messages, warnings, and errors produced by each class in DASSH
"""
# Notes on logging levels
# (https://stackoverflow.com/questions/2031163/when-to-use-the-different-log-levels)
# Trace:    Only when I would be "tracing" the code and trying to find
#           one part of a function specifically.
# Debug:    Information that is diagnostically helpful to people more
#           than just developers (IT, sysadmins, etc.).
# Info:     Generally useful information to log (service start/stop,
#           configuration assumptions, etc). Info I want to always have
#           available but usually don't care about under normal
#           circumstances. This is my out-of-the-box config level.
# Warn:     Anything that can potentially cause application oddities,
#           but for which I am automatically recovering. (Such as
#           switching from a primary to backup server, retrying an
#           operation, missing secondary data, etc.)
# Error:    Any error which is fatal to the operation, but not the
#           service or application (can't open a required file, missing
#           data, etc.). These errors will force user (administrator,
#           or direct user) intervention. These are usually reserved
#           (in my apps) for incorrect connection strings, missing
#           services, etc.
# Critical: Any error that is forcing a shutdown of the service or
#   `       application to prevent data loss (or further data loss).
#           I reserve these only for the most heinous errors and
#           situations where there is guaranteed to have been data
#   `       corruption or loss.
########################################################################
import logging
import os
import sys

LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = LOG_LEVEL - 5


def init_root_logger(path, name):
    # Register a new log level
    logging.addLevelName(FILE_LOG_LEVEL, "INFO_FILE")
    # Create the Logger
    logger = logging.getLogger(name)
    logger.setLevel(FILE_LOG_LEVEL)

    # Create a Formatter for formatting the log messages
    # log_file_formatter = logging.Formatter('%(asctime)s - '
    #                                        '%(levelname)s - '
    #                                        '%(message)s',
    #                                        datefmt='%d-%b-%y %H:%M:%S')
    log_file_formatter = logging.Formatter('%(asctime)s - '
                                           '%(name)18s - '
                                           '%(levelname)8s - '
                                           '%(message)s - ',
                                           datefmt='%d-%b-%y %H:%M:%S')
    # Create the Handler for logging data to a file
    logger_file_handler = logging.FileHandler(
        f'{os.path.join(path, name.lower())}.log',
        'w+')
    logger_file_handler.setLevel(FILE_LOG_LEVEL)

    # Add the Formatter to the Handler
    logger_file_handler.setFormatter(log_file_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_file_handler)

    # Repeat the above for a to-screen logger
    logger_stream_handler = logging.StreamHandler()
    logger_stream_handler.setLevel(LOG_LEVEL)
    logger_stream_formatter = \
        logging.Formatter(f'{name.upper()}....%(message)s')
    logger_stream_handler.setFormatter(logger_stream_formatter)
    logger.addHandler(logger_stream_handler)
    # Write to the streams so that the root logger is fully configured
    logger.info(f"{name.upper()} logger initialized")

    return logger


def init_logger(name):
    # This is used here and in other functions to abstract the specific
    # logger used
    return logging.getLogger(name)


class LoggedClass(object):
    """This class provides a consistent logger interface across
    classes.

    It should be inherited from for all classes that want a logger,
    and the class should call the LoggedClass' init method to initialize
    the logger."""

    def __init__(self, default_indent, name):
        self._default_indent = default_indent
        self._logger = init_logger(name)

    def log(self, level, message, indent=None):
        # Classes call this to write a log
        if indent is None:
            use_indent = self._default_indent
        else:
            use_indent = indent

        if level in ["error", "critical"]:
            use_indent = 0

        low_level = level.lower()
        if low_level in ["info", "warning", "error", "critical", "debug"]:
            func = getattr(self._logger, low_level)
            func(use_indent * " " + message)
            if low_level in ["error", "critical"]:
                sys.exit(1)
        elif low_level in ["info_file"]:
            self._logger.log(FILE_LOG_LEVEL, use_indent * " " + message)
        else:
            raise ValueError("Invalid level: {}".format(low_level))


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    From: https://stackoverflow.com/a/60462619
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeFilter(self)
