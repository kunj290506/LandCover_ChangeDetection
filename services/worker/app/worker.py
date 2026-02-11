import os
from redis import Redis
from rq import Worker, Queue, Connection

from landcover_common.settings import Settings


def main():
    settings = Settings()
    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        worker = Worker([Queue("batch")])
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
