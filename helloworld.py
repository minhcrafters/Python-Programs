from string import ascii_letters
import time
import random


def test1(target: str):
    result = [" "] * len(target)
    i = 0

    for e in target:
        if e == " ":
            i += 1
            continue
        for x in ascii_letters:
            result[i] = x

            print("\033c", end="")
            print("".join(result))

            if e == x:
                i += 1
                break
            time.sleep(0.04)


def test2(target: str, speed: int):
    for char in target:
        print(char, end="", flush=True)
        time.sleep(speed)


def test3(target: str, speed: int):
    for char in reversed(target):
        print(char, end="", flush=True)
        time.sleep(speed)


def test4(target: str):
    for char in target:
        # you_decide = random.randint(0, 1)
        # if you_decide == 1:
        #     char = char.upper()
        # else:
        #     char = char.lower()
        print(char, end="", flush=True)
        time.sleep(random.uniform(0.08, 0.4))


def test5(target: str):
    targets = target.split("\n")
    for string in targets:
        test4(string)
        time.sleep(random.uniform(0.005, 0.075))
        print()
        time.sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    # test1("Hello world")
    # test2("Hello world", 0.05)
    # print()
    # test3("Hello world", 0.05)
    # print()
    # test4("Hello world")
    # print()
    test5("""hello viewers
today i will show you how to code in python
type this
print("hello world")
thanks for watching
bye""")
