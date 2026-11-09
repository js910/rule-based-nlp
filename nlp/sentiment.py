"""KNU 한국어 감성사전과 도메인 특화 표현으로 감성 점수를 계산한다."""

import csv
import json
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
    r"합니다|이라|이라서|이고|진|적|적이다|적이에요|더라|더라고요|었음|았음|"
    r"네|었네|았네|였다|였어요|이었다|이었어요|던데|었다면|았다면|지네|지네요|지다|지요|임)?"
)
# '-지 않다', '-지 못하다'에서 뒷어절의 활용형을 식별한다.
POST_NEGATION = re.compile(r"(?:않|못하|못해|못했)(?:[가-힣]*)")
# 명시적으로 지원하는 이중부정은 '-지 않지 않다'의 활용형이다.
DOUBLE_FIRST = {"않지", "않지는", "않지도"}
PRE_NEGATION = {"안", "못"}
NEGATING_END = {"지", "지가", "지는", "지도", "진", "하지", "하지는", "하지도"}
IRREGULAR_FORMS = {"나빠": "나쁘", "예뻐": "예쁘", "느려": "느리", "질겨": "질기",
                   "고마워": "고맙", "쉬워": "쉽", "어려워": "어렵", "아파": "아프",
                   "지겨워": "지겹", "따뜻해": "따뜻", "아까워": "아깝", "부러워": "부럽",
                   "귀여워": "귀엽", "우스워": "우습", "즐거워": "즐겁", "재밌": "재미있",
                   "잼있": "재미있"}
INTENSIFIERS = {"매우": 1.5, "정말": 1.5, "아주": 1.5, "너무": 1.5,
                "무척": 1.5, "진짜": 1.5, "완전": 2.0, "엄청": 2.0,
                "굉장히": 2.0, "조금": 0.5, "약간": 0.5}


def load_lexicon():
    data = Path(__file__).resolve().parents[1] / "data"
    path = data / "SentiWord_info.json"
    if not path.exists():
        raise SystemExit("KNU 감성사전이 필요합니다. 먼저 python setup_data.py를 실행하세요.")
    with path.open(encoding="utf-8-sig") as source:
        entries = json.load(source)

    roots = {}
    words = {}
    phrases = {}
    for entry in entries:
        word, root = entry["word"], entry["word_root"]
        score = int(entry["polarity"])
        parts = word.split()
        if not score or not re.fullmatch(r"[가-힣]+(?:\s+[가-힣]+)*", word):
            continue
        # 여러 어절의 표현은 원문 그대로 비교한다. 일반 부정·강조 구문은 규칙에 맡긴다.
        if len(parts) > 1:
            if not any(part in PRE_NEGATION or part in INTENSIFIERS
                       or POST_NEGATION.fullmatch(part) for part in parts):
                phrases[" ".join(parts)] = score
            continue
        words[word] = score
        if root and word.startswith(root) and SUFFIX.fullmatch(word[len(root):]):
            roots.setdefault(root, score)
    roots.update(words)
    with (data / "domain_lexicon.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            term = row["term"]
            target = phrases if " " in term else roots
            target[term] = int(row["score"])
    return roots, phrases


LEXICON, PHRASES = load_lexicon()
MAX_TERM_LENGTH = max(map(len, LEXICON))
PHRASES_BY_START = {}
for phrase in PHRASES:
    parts = tuple(phrase.split())
    PHRASES_BY_START.setdefault(parts[0], []).append(parts)
for candidates in PHRASES_BY_START.values():
    candidates.sort(key=len, reverse=True)


def lookup(token):
    if token in INTENSIFIERS or token in PRE_NEGATION or POST_NEGATION.fullmatch(token):
        return None
    for length in range(min(len(token), MAX_TERM_LENGTH), 0, -1):
        stem = token[:length]
        if stem in LEXICON and SUFFIX.fullmatch(token[length:]):
            return stem, token[length:]
        irregular = IRREGULAR_FORMS.get(stem)
        if irregular in LEXICON and SUFFIX.fullmatch(token[length:]):
            return irregular, token[length:]
    return None


def analyze(text, use_rules=True, zero_label="중립"):
    tokens = tuple(TOKEN.findall(text))
    hits = []
    next_index = 0
    for index, token in enumerate(tokens):
        if index < next_index:
            continue
        phrase = next((parts for parts in PHRASES_BY_START.get(token, ())
                       if tokens[index:index + len(parts)] == parts), ())
        if phrase:
            stem, suffix = " ".join(phrase), ""
            base = PHRASES[stem]
            next_index = index + len(phrase)
        else:
            match = lookup(token)
            if match is None:
                continue
            stem, suffix = match
            base = LEXICON[stem]
            next_index = index + 1
        before = tokens[index - 1] if index else ""
        after = tokens[next_index:next_index + 2]
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
        hits.append({"token": " ".join(tokens[index:next_index]), "term": stem, "base": base,
                     "negations": negations, "weight": weight,
                     "score": base * (-1) ** negations * weight})

    score = sum(hit["score"] for hit in hits)
    label = "긍정" if score > 0 else "부정" if score < 0 else zero_label
    return {"label": label, "score": score, "tokens": list(tokens), "hits": hits}
