"""sources.yaml 에 적힌 데이터셋을 raw/<name>/ 아래로 내려받는다.

    python download.py --sources sources.yaml
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

from common import ROOT

RAW_DIR = ROOT / "raw"


def download_roboflow(src: dict, dest: Path) -> None:
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY 환경변수가 필요합니다 (roboflow.com 무료 계정 → Settings → API Keys).")
    project = Roboflow(api_key=api_key).workspace(src["workspace"]).project(src["project"])
    project.version(src["version"]).download("yolov8", location=str(dest), overwrite=True)


def download_url(src: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "download.zip"
    print(f"  {src['url']} 다운로드 중…")
    urllib.request.urlretrieve(src["url"], archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()
    # Mendeley 처럼 zip 안에 zip 이 또 들어 있는 경우 풀어 준다
    for inner in list(dest.rglob("*.zip")):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        inner.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=ROOT / "sources.yaml", type=Path)
    parser.add_argument("--only", nargs="*", help="이 이름의 출처만 받기")
    args = parser.parse_args()

    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["sources"]
    for src in sources:
        if args.only and src["name"] not in args.only:
            continue
        dest = RAW_DIR / src["name"]
        print(f"[{src['name']}] ({src['type']})")
        if src["type"] == "roboflow":
            download_roboflow(src, dest)
        elif src["type"] == "url":
            if dest.exists():
                shutil.rmtree(dest)
            download_url(src, dest)
        elif src["type"] == "local":
            print(f"  로컬 폴더 사용: {src['path']}")
        else:
            raise SystemExit(f"알 수 없는 type: {src['type']}")


if __name__ == "__main__":
    main()
