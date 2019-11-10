import threading
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy

from comet_ml import Experiment


def new_experiment(disabled=False):
    return Experiment(
        api_key="lzBiJi5UcZCwhAqlFNCUP4Qpg",
        project_name="athena",
        workspace="skr-io7803",
        disabled=disabled,
    )


def new_rpc_experiment_logger(experiment: Experiment, host='localhost', port=8085):
    methods = [i for i in dir(experiment) if callable(getattr(experiment, i))]
    log_methods = [i for i in methods if i.startswith('log_')]

    server = SimpleXMLRPCServer((host, port), allow_none=True)

    def make_handler(method):
        def handler(args, kwargs):
            getattr(experiment, method)(*args, **kwargs)

        return handler

    for method in log_methods:
        server.register_function(make_handler(method), method)

    def client_generator():
        return ExperimentProxy(host, port)

    def worker():
        server.serve_forever()

    threading.Thread(target=worker).start()

    return server, client_generator


class ExperimentProxy(object):

    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._setup()

    def _setup(self):
        self._proxy = ServerProxy('http://{}:{}'.format(self._host, self._port), allow_none=True)

    def __setstate__(self, state):
        self._host = state['host']
        self._port = state['port']
        self._setup()

    def __getstate__(self):
        return {
            'host': self._host,
            'port': self._port,
        }

    def __getattr__(self, item):
        if item in ('__setstate__', '__getstate__', '_setup'):
            return getattr(self, item)

        attr = getattr(self._proxy, item)

        def proxy_caller(*args, **kwargs):
            return attr(args, kwargs)

        if item.startswith('log_'):
            return proxy_caller

        return attr
