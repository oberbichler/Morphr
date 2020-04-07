from colorama import init, Fore, Style
from typing import List, Optional
from pydantic import BaseModel


class Logger:
    def __init__(self, info_level=1):
        self._info_levels = [info_level]

    @property
    def info_level(self):
        return self._info_levels[-1]

    def push(self, info_level):
        if info_level is None:
            return
        self._info_levels.append(info_level)

    def pop(self):
        if len(self._info_levels) > 1:
            self._info_levels.pop()

    def h1(self, text):
        if self.info_level < 1:
            return
        print(Fore.GREEN + Style.BRIGHT + text + Style.RESET_ALL)

    def h2(self, text):
        if self.info_level < 1:
            return
        print(Fore.YELLOW + Style.BRIGHT + text + Style.RESET_ALL)

    def info(self, text):
        if self.info_level < 2:
            return
        print(text)
