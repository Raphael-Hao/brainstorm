#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /build.py
# \brief:
# Author: raphael hao
import argparse
import pathlib
import subprocess

import yaml


def get_build_args():
    parser = argparse.ArgumentParser(description="Build docker image")
    parser.add_argument(
        "--config",
        type=str,
        default="gh_config.yaml",
        help="Config file for docker build",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.absolute()
    config_file = script_dir / str(args.config)
    config = yaml.safe_load(config_file.open())

    args.type = config["type"]
    args.username = config["username"]
    args.token = config["token"]
    args.base_image = config["base_image"]
    args.tag = config["tag"]
    args.branch = config["branch"]
    args.upload = config["upload"]
    args.updata_brt = config["update_brt"]
    args.no_cache = config["no_cache"]
    args.context: str = config["context"]
    context_dir = (
        script_dir if args.context is None else script_dir / pathlib.Path(args.context)
    )
    args.context_path = context_dir.as_posix()
    args.dockerfile = (script_dir / f"../docker/Dockerfile.{args.type}").as_posix()
    args.image_spec = f"brt:{args.branch}" if args.tag is None else f"brt:{args.tag}"

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
    subprocess.call(az_acr_login_cmd)


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
    if args.type == "update":
        cmd.extend(["--build-arg", f"UPDATE_BRT_ONLY={args.update_brt}"])
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
    subprocess.call(cmd)

    if args.upload:
        login_github_registry(args.username, args.token)
        docker_upload(args.image_spec, args.username)


if __name__ == "__main__":
    build_docker()
