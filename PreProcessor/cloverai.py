import re

def is_spam(text):
    strong_keywords = ["몰카", "키스방", "오피", "토토", "건마 ", "풀싸롱", "립카페", "핸플"]
    weak_keywords = ["스와핑","엘프녀","회원가입","무료","채팅", "안마", "국산","셀카", "매직미러","op", "야동","만남", "카페", "비아그라","중년여성", "채팅어플", "탈의실", "실시간"]

    for keyword in strong_keywords:
        if keyword in text:
            return True

    count = 0
    for keyword in weak_keywords:
        if keyword in text:
            count += 1

    if count > 2 :
        return True

    return False


def split_sentence(text):
    reg = """(?x)[^.!?…\s]   # First char is non-punct, non-ws
      [^.!?…]*         # Greedily consume up to punctuation.
      (?:              # Group for unrolling the loop.
        [.!?…]         # (special) inner punctuation ok if
        (?!['\"]?\s|$) # not followed by ws or EOS.
        [^.!?…]*       # Greedily consume up to punctuation.
      )*               # Zero or more (special normal*)
      [.!?…]?          # Optional ending punctuation.
      ['\"]?           # Optional closing quote.
      (?=\s|$)"""

    r = re.compile(reg)

    arr = []
    for match in r.finditer(text):
        arr.append(match.group(0))

    return arr

