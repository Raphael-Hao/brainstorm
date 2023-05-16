# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pathlib
import subprocess
import sys


def get_build_args():
    parser = argparse.ArgumentParser(description="Build docker image")
    parser.add_argument(
        "--base",
        type=str,
        choices=["nvidia", "brt"],
        default="update",
        help="Type of base image to build on top of, can be nvidia or brt",
    )
    parser.add_argument("--branch", type=str, default="main", help="BRT branch to use")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cache")
    parser.add_argument(
        "--update-brt-only",
        action="store_true",
        help="Update BRT only",
    )
    parser.add_argument("--upload", action="store_true", help="Upload to github")
    parser.add_argument(
        "--username",
        type=str,
        required="--upload" in sys.argv,
        default=None,
        help="Github username",
    )
    parser.add_argument(
        "--token",
        type=str,
        required="--upload" in sys.argv,
        default=None,
        help="Github token",
    )

    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.absolute()

    if args.base == "nvidia":
        args.base_image = "nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04"
    elif args.base == "brt":
        args.base_image = f"ghcr.io/raphael-hao/brt:{args.branch}"
    args.context_path = script_dir.as_posix()
    dockerfile_suffix = "base" if args.base == "nvidia" else "update"
    args.dockerfile = (
        script_dir / f"../../docker/Dockerfile.{dockerfile_suffix}"
    ).as_posix()
    args.image_spec = f"brt:{args.branch}"

    return args


def login_github_registry(username, token):
    az_acr_login_cmd = [
        "echo",
        token,
        "|",
        "docker",
        "login",
        "ghcr.io",
        "-u",
        username,
        "--password-stdin",
    ]
    print("Logging into github registry...")
    az_acr_login_cmd_str = " ".join(az_acr_login_cmd)
    print(az_acr_login_cmd_str)
    login_output = subprocess.getoutput(az_acr_login_cmd_str)
    print(login_output)


def docker_upload(image_spec, username):
    docker_tag_cmd = ["docker", "tag", image_spec]
    new_tag = f"ghcr.io/{username}/{image_spec}"
    print(f"Tagging image {image_spec} with tag: {new_tag}")
    docker_tag_cmd.append(new_tag)
    subprocess.call(docker_tag_cmd)
    docker_push_cmd = ["docker", "push", new_tag]
    print(f"Pushing image: {new_tag} to github registry")
    subprocess.call(docker_push_cmd)


def build_docker():
    args = get_build_args()
    cmd = [
        "docker",
        "build",
        "-t",
        args.image_spec,
    ]

    cmd.extend(["--build-arg", f"BASE_IMAGE={args.base_image}"])
    cmd.extend(["--build-arg", f"BRT_BRANCH={args.branch}"])
    if args.base == "brt":
        cmd.extend(["--build-arg", f"UPDATE_BRT_ONLY={args.update_brt_only}"])
        cmd.append("--no-cache")
    if args.no_cache and "--no-cache" not in cmd:
        cmd.extend(["--no-cache"])
    cmd.extend(["-f", args.dockerfile])
    cmd.extend(["--progress=plain", args.context_path])
    print(" ".join(cmd))
    print("Building docker image...")
    print(f"Image spec: {args.image_spec}")
    print(f"Base image: {args.base_image}")
    print(f"Using Branch: {args.branch} of BRT")
    print(f"Using Dockerfile: {args.dockerfile}")
    print(f"Using context: {args.context_path}")
    print(cmd)
    subprocess.call(cmd)

    if args.upload:
        login_github_registry(args.username, args.token)
        docker_upload(args.image_spec, args.username)


if __name__ == "__main__":
    build_docker()
