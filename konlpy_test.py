# -*- coding: euc-kr -*-
from konlpy.tag import Kkma
from konlpy.tag import Twitter


lib = Twitter()

print(1)
print(lib.tokenize("ǥ���Ͻô� ��ü�� �̹� ���ڰ� �����Ͻô°� ����� ǥ���Ͻô� �� �����ϴ�."))
print(2)
print(lib.pos("����  ���� ���� ���� �߽��ϴ�..!!!"))
print(3)
print(lib.pos(" "))