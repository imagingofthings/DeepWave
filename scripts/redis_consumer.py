import argparse
import logging
import pathlib
import subprocess
import tempfile

import redis


def parse_args():
    parser = argparse.ArgumentParser(description='Consume tasks from Redis work queue.')
    parser.add_argument('--hostname',
                        help='Redis hostname.',
                        required=True,
                        type=str)
    parser.add_argument('--port',
                        help='Redis port.',
                        required=True,
                        type=int)
    parser.add_argument('--queue',
                        help='Redis key from which to pull data.',
                        required=True,
                        type=str)
    parser.add_argument('--log_dir',
                        help=('Folder in which to store log file. '
                              "The log file's name is generated at random."),
                        required=True,
                        type=str)

    args = parser.parse_args()
    args.log_dir = pathlib.Path(args.log_dir).expanduser().absolute()
    if not (args.log_dir.exists() and args.log_dir.is_dir()):
        raise ValueError(f'Cannot log to "{args.log_dir}".')

    return args


def setup_log(args):
    regen_name = True
    while regen_name:
        log_fname = (args.log_dir /
                     (pathlib.Path(tempfile.NamedTemporaryFile().name).stem + '.log'))

        if not log_fname.exists():
            regen_name = False

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s | %(message)s',
                        filename=log_fname,
                        filemode='w')


def process(args):
    r = redis.Redis(host=args.hostname, port=args.port, db=0, decode_responses=True)

    continue_popping = True
    while continue_popping:
        cmd = r.lpop(args.queue)

        if cmd == None:  # Queue empty
            logging.info(f'No messages left on {args.hostname}:{args.port}:{args.queue}')
            logging.info('QUIT')
            continue_popping = False
        else:  # Execute command
            logging.info(f'START "{cmd}"')
            handle = subprocess.run([cmd], shell=True, stderr=subprocess.PIPE)
            if handle.returncode != 0:  # Something went wrong
                logging.info(f'Command returned non-zero error code {handle.returncode}')
                logging.info('sys.stderr')
                logging.info('----------')
                logging.info(handle.stderr)
            logging.info(f'END   "{cmd}"')


if __name__ == '__main__':
    args = parse_args()
    setup_log(args)
    process(args)
