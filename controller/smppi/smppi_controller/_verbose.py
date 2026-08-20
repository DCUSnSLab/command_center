"""라이브러리 초기화 출력 게이트.

크리틱·옵티마이저가 init 마다 print() 로 설정을 쏟아내면 런치 로그가 묻힌다.
기본은 조용하고, 필요할 때만 환경변수로 켠다:

    SMPPI_VERBOSE=1 ros2 launch ...
"""
import os
import sys

_ON = os.environ.get("SMPPI_VERBOSE", "").strip() not in ("", "0", "false", "False")


def vprint(*args, **kwargs):
    """SMPPI_VERBOSE 가 켜져 있을 때만 출력."""
    if _ON:
        kwargs.setdefault("file", sys.stderr)
        print(*args, **kwargs)
