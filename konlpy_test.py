# -*- coding: euc-kr -*-
from konlpy.tag import Kkma
from konlpy.tag import Twitter


lib = Twitter()

print(1)
print(lib.tokenize("표시하시는 자체가 이미 나쁘게 생각하시는거 대놓고 표현하시는 것 같습니다."))
print(2)
print(lib.pos("저도  전족 자주 생각 했습니다..!!!"))
print(3)
print(lib.pos(" "))