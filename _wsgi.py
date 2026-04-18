import os
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
        "propagate": True
    }
})

from label_studio_ml.api import init_app
from model import YOLOBackend

app = init_app(model_class=YOLOBackend)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label Studio ML Backend')
    parser.add_argument('-p', '--port', type=int, default=int(os.environ.get("PORT", 9090)),
                        help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)