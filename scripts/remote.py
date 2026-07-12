#!/usr/bin/env python
"""Minimal SSH/SFTP helper for running GPU experiments on the remote server.

Credentials come from the environment (never hard-coded / committed):

    RWL_SSH_HOST   (default: connect.westb.seetacloud.com)
    RWL_SSH_PORT   (default: 49844)
    RWL_SSH_USER   (default: root)
    RWL_SSH_PASS   (required)

Run with the *system* Python (has paramiko): ``C:/Python314/python.exe scripts/remote.py ...``

Subcommands:
    run "<cmd>"                 run a shell command on the server, stream stdout/stderr
    put <local> <remote>        upload a file (remote path is RELATIVE to the SFTP root /root)
    get <remote> <local>        download a file
    putdir <local> <remote>     upload a directory tree
"""

from __future__ import annotations

import os
import sys
import stat
from pathlib import Path

import paramiko


def _client() -> paramiko.SSHClient:
    host = os.environ.get("RWL_SSH_HOST", "connect.westb.seetacloud.com")
    port = int(os.environ.get("RWL_SSH_PORT", "49844"))
    user = os.environ.get("RWL_SSH_USER", "root")
    pw = os.environ.get("RWL_SSH_PASS")
    if not pw:
        sys.exit("RWL_SSH_PASS not set in environment")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw, timeout=30,
                banner_timeout=30, auth_timeout=30)
    return cli


def run(cmd: str) -> int:
    cli = _client()
    try:
        stdin, stdout, stderr = cli.exec_command(cmd, get_pty=False, timeout=None)
        for line in iter(stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
        err = stderr.read().decode(errors="replace")
        if err.strip():
            sys.stderr.write(err)
        return stdout.channel.recv_exit_status()
    finally:
        cli.close()


def _sftp(cli):
    return cli.open_sftp()


def put(local: str, remote: str) -> int:
    cli = _client()
    try:
        sftp = _sftp(cli)
        _mkdirs(sftp, str(Path(remote).parent).replace("\\", "/"))
        sftp.put(local, remote)
        print(f"put {local} -> {remote}")
        return 0
    finally:
        cli.close()


def get(remote: str, local: str) -> int:
    cli = _client()
    try:
        sftp = _sftp(cli)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, local)
        print(f"get {remote} -> {local}")
        return 0
    finally:
        cli.close()


def _mkdirs(sftp, remote_dir: str) -> None:
    if not remote_dir or remote_dir in (".", "/"):
        return
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}" if cur else p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def putdir(local: str, remote: str) -> int:
    cli = _client()
    try:
        sftp = _sftp(cli)
        local_root = Path(local)
        for path in local_root.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(local_root).as_posix()
            rpath = f"{remote}/{rel}"
            _mkdirs(sftp, str(Path(rpath).parent).replace("\\", "/"))
            sftp.put(str(path), rpath)
            print(f"put {rel}")
        return 0
    finally:
        cli.close()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    op = sys.argv[1]
    if op == "run":
        return run(sys.argv[2])
    if op == "put":
        return put(sys.argv[2], sys.argv[3])
    if op == "get":
        return get(sys.argv[2], sys.argv[3])
    if op == "putdir":
        return putdir(sys.argv[2], sys.argv[3])
    print(f"unknown op: {op}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
