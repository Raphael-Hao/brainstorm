# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from distutils.dist import Distribution
import subprocess
import os
import sys
import pathlib


def is_ninja_available():
    r"""
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    """
    try:
        subprocess.check_output("ninja --version".split())
    except Exception:
        return False
    else:
        return True


def get_build_dir_with_tmp():
    # pylint: disable=no-member
    # get build tmp dir from distutils build command
    dist = Distribution()
    build_cmd = dist.get_command_obj("build")
    build_cmd.ensure_finalized()
    build_tmp_dir = build_cmd.build_temp
    # check if build tmp dir exists
    if not os.path.exists(build_tmp_dir):
        raise RuntimeError(
            "build tmp dir does not exist, please run `pip install -v -e` . first"
        )
    return pathlib.Path(build_cmd.build_base), pathlib.Path(build_cmd.build_temp)


def create_compile_commands_json():
    if not is_ninja_available():
        raise RuntimeError(
            "ninja is not available for creating compile_commands.json, please install ninja first"
        )
    build_dir, build_tmp_dir = get_build_dir_with_tmp()
    compdb_cmd = ["ninja", "-C"]
    compdb_cmd.extend([build_tmp_dir.as_posix()])
    compdb_cmd.extend(["-t", "compdb"])
    compdb_filepath = build_dir / "compile_commands.json"
    compdb_f = compdb_filepath.open("w")
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        subprocess.run(
            compdb_cmd,
            stdout=compdb_f,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = "Generating compile_commands.json failed with error: "
        # `error` is a CalledProcessError (which has an `ouput`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, "output") and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*())}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


create_compile_commands_json()
