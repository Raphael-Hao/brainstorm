import pathlib

__all__ = ["test_path"]
test_path = pathlib.Path("/home/whcui/brainstorm_project/brainstorm/sample/miscellanea")
test_path.mkdir(parents=True, exist_ok=True)
