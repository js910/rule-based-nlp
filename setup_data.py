"""공식 저장소의 고정 버전에서 평가 및 감성 분석 데이터를 받는다."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).parent / "data"
FILES = {
    "SentiWord_info.json": (
        "https://raw.githubusercontent.com/park1200656/KnuSentiLex/"
        "c95a8a9bcb78ef92e3f8ddd277abaf31451d9f23/data/SentiWord_info.json",
        "a455872e74a18330ea7c7938402acd236eb1d3efbc6f3ffe3510ba0503feae20",
    ),
    "ratings_test.txt": (
        "https://raw.githubusercontent.com/e9t/nsmc/"
        "eafdd77e310f399a4d0083a58ade40b55a5a308d/ratings_test.txt",
        "8ac9f64052f11dbf6ae0acb5e038f03d90a76f0eda7820cfb3a92d02edfcebda",
    ),
}


def main():
    for name, (url, expected_hash) in FILES.items():
        path = DATA / name
        if path.exists() and sha256(path.read_bytes()).hexdigest() == expected_hash:
            print(f"{name}: 준비 완료")
            continue
        with urlopen(url, timeout=30) as response:
            content = response.read()
        if sha256(content).hexdigest() != expected_hash:
            raise SystemExit(f"{name}: 다운로드 파일의 SHA-256이 다릅니다.")
        temp = path.with_suffix(".download")
        temp.write_bytes(content)
        temp.replace(path)
        print(f"{name}: 다운로드 완료")


if __name__ == "__main__":
    main()
