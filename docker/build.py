#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /build.py
# \brief:
# Author: raphael hao
import pathlib
import argparse
import subprocess


def get_build_args():
    parser = argparse.ArgumentParser(description="Build docker image")
    parser.add_argument(
        "--type",
        type=str,
        default="sing",
        choices=["sing", "update", "msra"],
        help="Build type",
    )
    parser.add_argument(
        "--repository",
        type=str,
        default="brainstorm",
        help="Repository name under container registry",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04",
        help="Base image to build",
    )
    parser.add_argument("--remote", action="store_true", help="Using remote image")
    parser.add_argument("--tag", type=str, default=None, help="Container image tag")
    parser.add_argument("--branch", type=str, default="main", help="Branch name")
    parser.add_argument("--upload", action="store_true", help="Upload to azure")
    parser.add_argument(
        "--registry", type=str, default=None, help="Container image registry"
    )
    parser.add_argument("--context", type=str, default=None, help="Context path")
    parser.add_argument("--ssh-key", type=str, default="id_ed25519", help="SSH key")
    parser.add_argument("--update-brt", action="store_false", help="Only update brt")
    args = parser.parse_args()
    if args.remote or args.upload:
        assert (
            args.registry is not None
        ), "Registry name is required for remote base image or upload"
    if args.remote:
        args.base_image = f"{args.registry}/{args.base_image}"
    script_dir = pathlib.Path(__file__).parent.absolute()
    context_dir = script_dir if args.context is None else pathlib.Path(args.context)
    args.context_path = context_dir.as_posix()
    args.dockerfile = (context_dir / f"Dockerfile.{args.type}").as_posix()
    args.image_spec = f"brt:{args.branch}" if args.tag is None else f"brt:{args.tag}"

    return args


def az_acr_login(registry):
    az_acr_login_cmd = ["az", "acr", "login", "--name", registry]
    subprocess.call(az_acr_login_cmd)


def az_acr_get_image_tag(image_repo, registry):
    get_image_tag_cmd = [
        "az",
        "acr",
        "repository",
        "show-manifests",
        "--name",
        registry,
        "--repository",
        image_repo,
        "--orderby",
        "time_desc",
        "--query",
        """[].{Tag:tags[0]}""",
        "--output",
        "tsv",
        "--top",
        "1",
    ]
    return subprocess.check_output(get_image_tag_cmd, encoding="UTF-8").strip()


def get_singularity_image():
    singularity_registry = "singularitybase"
    validator_image_repo = "validations/base/singularity-tests"
    installer_image_repo = "installer/base/singularity-installer"
    az_acr_login_cmd = ["az", "acr", "login", "--name", singularity_registry]
    subprocess.call(az_acr_login_cmd)
    validator_image_tag = az_acr_get_image_tag(
        validator_image_repo, singularity_registry
    )
    installer_image_tag = az_acr_get_image_tag(
        installer_image_repo, singularity_registry
    )
    validtaor_image = f"{singularity_registry}.azurecr.io/{validator_image_repo}:{validator_image_tag}"
    installer_image = f"{singularity_registry}.azurecr.io/{installer_image_repo}:{installer_image_tag}"
    return validtaor_image, installer_image


def az_acr_upload(image_spec, registry, repository=None):
    az_acr_login(registry)
    docker_tag_cmd = ["docker", "tag", image_spec]
    new_tag = f"{registry}.azurecr.io/{image_spec}"
    print(f"Tagging image {image_spec} to {new_tag}")
    if repository is not None:
        new_tag = f"{registry}.azurecr.io/{repository}/{image_spec}"
    docker_tag_cmd.append(new_tag)
    subprocess.call(docker_tag_cmd)
    docker_push_cmd = ["docker", "push", new_tag]
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
    cmd.extend(["--build-arg", f"SSH_KEY_FILE={args.ssh_key}"])
    cmd.extend(["--build-arg", f"BRANCH={args.branch}"])
    if args.type == "sing":
        validator_image, installer_image = get_singularity_image()
        cmd.extend(["--build-arg", f"VALIDATOR_IMAGE={validator_image}"])
        cmd.extend(["--build-arg", f"INSTALLER_IMAGE={installer_image}"])
    if args.type == "update":
        cmd.extend(["--build-arg", f"UPDATE_BRT_ONLY={args.update_brt}"])
    cmd.extend(["-f", args.dockerfile])
    cmd.extend(["--progress=plain", args.context_path])
    print(" ".join(cmd))
    print("Building docker image...")
    print(f"Image spec: {args.image_spec}")
    print(f"Base image: {args.base_image}")
    print(f"Using Branch: {args.branch} of BRT")
    print(f"Using SSH key: {args.ssh_key}")
    print(f"Using Dockerfile: {args.dockerfile}")
    print(f"Using context: {args.context_path}")
    print(f"Using registry: {args.registry}")
    if args.type == "sing":
        print(f"Using validator image: {validator_image}")
        print(f"Using installer image: {installer_image}")
    subprocess.call(cmd)

    if args.upload:
        az_acr_upload(args.image_spec, args.registry, args.repository)


if __name__ == "__main__":
    build_docker()
