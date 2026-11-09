"""정규표현식으로 다섯 종류의 정보를 찾고 정규화한다."""

import re
from datetime import date
from urllib.parse import urlsplit, urlunsplit

# 로컬 파트의 점·플러스 태그와 여러 단계의 도메인을 허용한다.
# 예: user@example.com, user.name@sub.example.com, user+tag@example.org
EMAIL = re.compile(
    r"(?<![\w.+@-])(?P<local>[a-z0-9_%+-]+(?:\.[a-z0-9_%+-]+)*)"
    r"@(?P<domain>(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})"
    r"(?![a-z0-9_-]|\.[a-z0-9])",
    re.I | re.ASCII,
)

# 휴대폰, 서울, 세 자리 지역번호. 하이픈·공백·괄호와 붙여 쓰기를 허용한다.
# 한국 국가번호 +82와 하이픈으로 나뉜 한글 숫자 표기도 지원한다.
# 예: (02) 123-4567, +82 10-1234-5678, 공일공-일이삼사-오육칠팔
KOREAN_DIGITS = "공영일이삼사오육칠팔구"
PHONE_DIGITS = str.maketrans(KOREAN_DIGITS, "00123456789")
PHONE = re.compile(
    r"(?<![0-9])(?:(?:\+82[ -]?(?P<international_area>2|[13-6][0-9])"
    r"|\((?P<bracket>02|0[13-6][0-9])\)|(?P<area>02|0[13-6][0-9]))"
    r"[ -]?(?P<middle>[0-9]{3,4})[ -]?(?P<last>[0-9]{4})(?![0-9])"
    rf"|(?<![{KOREAN_DIGITS}0-9])(?P<spoken>[공영](?:이|[일삼사오육][{KOREAN_DIGITS}])"
    rf"-[{KOREAN_DIGITS}]{{3,4}}-[{KOREAN_DIGITS}]{{4}})(?![{KOREAN_DIGITS}0-9]))"
)

# 한국어 날짜와 연·월·일 순서의 숫자 날짜를 찾는다. 구분자는 역참조로 통일한다.
# 예: 2024년 2월 29일, 2024/02/29, 2024-02-29. 2024/02-29는 제외한다.
DATE = re.compile(
    r"(?<![0-9])(?P<year>[0-9]{4})"
    r"(?:년\s*(?P<km>[0-9]{1,2})월\s*(?P<kd>[0-9]{1,2})일"
    r"|(?P<sep>[-/.])(?P<nm>[0-9]{1,2})(?P=sep)(?P<nd>[0-9]{1,2}))"
    r"(?![0-9])"
)

# 숫자 묶음은 쉼표가 있으면 세 자리 단위만 허용한다.
NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
# 통화 기호나 원화 단위를 통해 일반 숫자와 금액을 구별한다.
# 예: 10,000원, 1억 2천만원, $100
MONEY = re.compile(
    rf"(?<![0-9.,])(?:(?P<symbol>[$₩])\s*(?P<amount>{NUMBER})(?![0-9]|[.,][0-9])"
    rf"|(?P<won>(?:{NUMBER}[조억만천백십]\s*)*"
    rf"(?:{NUMBER}|{NUMBER}[조억만천백십]|[만천백십])\s*)원)"
)

# 복합 원화에서 숫자와 단위가 이루는 각 항을 읽는다.
WON_TERM = re.compile(rf"({NUMBER})?([조억만천백십]?)")
SMALL = {"": 1, "십": 10, "백": 100, "천": 1000}
LARGE = {"만": 10_000, "억": 100_000_000, "조": 1_000_000_000_000}

# http, https, www 시작 주소와 선택적인 포트, 경로, 쿼리, 프래그먼트를 찾는다.
# URL의 경로는 ASCII 또는 퍼센트 인코딩 표기를 지원한다.
# 예: http://example.com, https://example.com/a?q=1, www.example.com
URL = re.compile(
    r"(?<![a-z0-9@._-])(?:(?:https?://)(?:www\.)?|www\.)"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}"
    r"(?::[0-9]{1,5})?(?:[/?#][a-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?"
    r"(?![a-z0-9_-]|\.[a-z0-9])",
    re.I | re.ASCII,
)

PATTERNS = {"email": EMAIL, "phone": PHONE, "date": DATE, "money": MONEY, "url": URL}


def parse_won(value):
    total = section = 0
    for digits, unit in WON_TERM.findall(value):
        if not digits and not unit:
            continue
        number = int(digits.replace(",", "")) if digits else 1
        if unit in LARGE:
            coefficient = section + (number if digits else 0)
            total += (coefficient if digits or section else 1) * LARGE[unit]
            section = 0
        else:
            section += number * SMALL[unit]
    return total + section


def normalize(kind, match):
    if kind == "email":
        return f"{match['local']}@{match['domain'].lower()}"
    if kind == "phone":
        if match["spoken"]:
            return match["spoken"].translate(PHONE_DIGITS)
        area = "0" + match["international_area"] if match["international_area"] else (
            match["bracket"] or match["area"])
        return "-".join((area, match["middle"], match["last"]))
    if kind == "date":
        return date(int(match["year"]), int(match["km"] or match["nm"]),
                    int(match["kd"] or match["nd"])).isoformat()
    if kind == "money":
        if match["symbol"]:
            currency = "USD" if match["symbol"] == "$" else "KRW"
            return {"amount": int(match["amount"].replace(",", "")), "currency": currency}
        return {"amount": parse_won(match["won"]), "currency": "KRW"}

    value = match.group().rstrip(".,;!?")
    for opening, closing in (("(", ")"), ("[", "]")):
        while value.endswith(closing) and value.count(closing) > value.count(opening):
            value = value[:-1]
    if "://" not in value:
        value = "https://" + value
    parts = urlsplit(value)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path, parts.query, parts.fragment))


def extract(text):
    """유형별로 정규화된 결과를 등장 순서대로 반환한다."""
    found = {kind: [] for kind in PATTERNS}
    occupied = []
    # URL 및 이메일 내부 문자열을 전화·금액·날짜로 중복 집계하지 않는다.
    for kind in ("url", "email", "phone", "date", "money"):
        for match in PATTERNS[kind].finditer(text):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            try:
                found[kind].append(normalize(kind, match))
            except ValueError:
                continue
            occupied.append(match.span())
    return found
