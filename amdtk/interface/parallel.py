
"""
Wrapper over the ipyparallel module.

The main purpose of this module is to provide a way to manage the
ipyparallel cluster within a python session..

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import logging
import subprocess
import time
from contextlib import contextmanager
from ipyparallel import Client


# Create the module's logger.
logger = logging.getLogger(__name__)

# Command to start the ipyparallel server.
START_SERVER_CMD = 'ipcluster start {profile} -n {njobs} --daemonize --quiet'


# Command to stop the ipyparallel server.
STOP_SERVER_CMD = 'ipcluster stop {profile} --quiet'


@contextmanager
def parallel(profile, njobs, delay=20):
    try:
        # It seems that ipyparallel add sneakily add another log
        # handler the root level. We remove it to make sure this will
        # not pollute our logs.
        rootLogger = logging.getLogger()
        nhandlers = len(rootLogger.handlers)

        # Start the ipyparallel server.
        logger.debug('starting ipyparallel server profile={profile}, '
                     'njobs={njobs}'.format(profile=profile, njobs=njobs))
        subprocess.run(START_SERVER_CMD.format(profile=profile,
                                               njobs=njobs), shell=True)

        # The server may be slow to create the clients. We give
        # it some time for this.
        logger.debug('waiting {delay} seconds for the server to '
                     'start'.format(delay=delay))
        time.sleep(delay)

        # Connect to the server.
        rc = Client(profile=profile)
        dview = rc[:]

        with dview.sync_imports():
            import theano

        if nhandlers < len(rootLogger.handlers):
            rootLogger.removeHandler(rootLogger.handlers[-1])

        logger.info('connected to {length} jobs'.format(length=len(dview)))
        yield dview
    finally:
        # Stop the server only if it has started successfully.
        logger.debug('shutting down the ipyparallel server')
        subprocess.run(STOP_SERVER_CMD.format(profile=profile), shell=True)

