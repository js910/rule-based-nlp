"""한국어 표면 토큰과 직접 구축한 사전을 이용한 감성 점수 계산."""

import csv
import re
from pathlib import Path

# 한글 어절, 영숫자 어절, 절 경계를 순서대로 보존한다.
TOKEN = re.compile(r"[가-힣]+|[a-zA-Z0-9]+|[.!?,;:。！？]")
# 어간 뒤에 허용하는 자주 쓰이는 활용형 및 명사 조사.
SUFFIX = re.compile(
    r"(?:다|아요|어요|요|네요|군요|습니다|입니다|이에요|예요|고|지만|는데|은데|"
    r"아서|어서|으면|으면요|았어요|었어요|았다|었다|았고|었고|았네요|었네요|"
    r"았지만|었지만|았다가|었다가|은|는|이|가|을|를|도|만|에|으로|에서|"
    r"지|지가|지는|지도|기|기가|기는|게|음|함|해|해요|해서|하다|하고|하지만|"
    r"하는|한|하지|하지는|하지도|하네요|했|했다|했어요|했고|했지만|했네요|"
    r"합니다|이라|이라서|이고)?"
)
# '-지 않다', '-지 못하다'에서 뒷어절의 활용형을 식별한다.
POST_NEGATION = re.compile(r"(?:않|못하|못해|못했)(?:[가-힣]*)")
# 명시적으로 지원하는 이중부정은 '-지 않지 않다'의 활용형이다.
DOUBLE_FIRST = {"않지", "않지는", "않지도"}
PRE_NEGATION = {"안", "못"}
NEGATING_END = {"지", "지가", "지는", "지도", "하지", "하지는", "하지도"}
IRREGULAR_FORMS = {"나빠": "나쁘", "예뻐": "예쁘", "느려": "느리", "질겨": "질기",
                   "고마워": "고맙", "쉬워": "쉽", "어려워": "어렵", "아파": "아프"}
INTENSIFIERS = {"매우": 1.5, "정말": 1.5, "아주": 1.5, "너무": 1.5,
                "무척": 1.5, "진짜": 1.5, "완전": 2.0, "엄청": 2.0,
                "굉장히": 2.0, "조금": 0.5, "약간": 0.5}


def load_lexicon():
    path = Path(__file__).parent / "data" / "lexicon.csv"
    with path.open(encoding="utf-8", newline="") as source:
        return {row["term"]: int(row["score"]) for row in csv.DictReader(source)}


LEXICON = load_lexicon()
MAX_TERM_LENGTH = max(map(len, LEXICON))


def lookup(token):
    for length in range(min(len(token), MAX_TERM_LENGTH), 0, -1):
        stem = token[:length]
        if stem in LEXICON and SUFFIX.fullmatch(token[length:]):
            return stem, token[length:]
        if stem in IRREGULAR_FORMS and SUFFIX.fullmatch(token[length:]):
            return IRREGULAR_FORMS[stem], token[length:]
    return None


def analyze(text, use_rules=True):
    tokens = TOKEN.findall(text)
    hits = []
    for index, token in enumerate(tokens):
        match = lookup(token)
        if match is None:
            continue
        stem, suffix = match
        before = tokens[index - 1] if index else ""
        after = tokens[index + 1:index + 3]
        negations = 0
        weight = 1.0
        if use_rules:
            if before in PRE_NEGATION:
                negations += 1
                before = tokens[index - 2] if index >= 2 else ""
            if suffix in NEGATING_END and after and POST_NEGATION.fullmatch(after[0]):
                negations += 1
                if after[0] in DOUBLE_FIRST and len(after) == 2 and POST_NEGATION.fullmatch(after[1]):
                    negations += 1
            weight = INTENSIFIERS.get(before, 1.0)
        base = LEXICON[stem]
        hits.append({"token": token, "term": stem, "base": base,
                     "negations": negations, "weight": weight,
                     "score": base * (-1) ** negations * weight})

    score = sum(hit["score"] for hit in hits)
    label = "긍정" if score > 0 else "부정" if score < 0 else "중립"
    return {"label": label, "score": score, "tokens": tokens, "hits": hits}
