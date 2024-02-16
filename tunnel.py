import time
import sys
import os
import navigate
import random

# school wifi doesn't allow me to install custom packages from pypi so yeah, enjoy my custom coded text-based game


shield = False
sword = False
room_order = 0


# I don't know why I need a progress bar
# def progressbar(iter, prefix="", size=60, out=sys.stdout):
#     count = len(iter)
#     start = time.time()
#     hide_cursor()

#     def show(j):
#         x = int(size * j / count)
#         remaining = ((time.time() - start) / j) * (count - j)

#         mins, sec = divmod(remaining, 60)
#         time_str = f"{int(mins):02}:{sec:05.2f}"

#         print(
#             f"{prefix}[{u'█' * x}{('.'*(size - x))}] {j}/{count} | ETA: {time_str}",
#             end="\r",
#             file=out,
#             flush=True,
#         )

#     for i, item in enumerate(iter):
#         yield item
#         show(i + 1)

#     show_cursor()
#     print("\n", flush=True, file=out)


def probably(chance: float):
    return random.random() < (chance / 100)


def printf(text: str, speed: float = 0.04):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(speed)
    time.sleep(0.5)
    print()


def get_choice(correct: list):
    choices = ["Choose an option:", *correct]
    return choices[
        navigate.select(
            choices,
            caption_indices=[0],  # deselected_prefix="  ", selected_prefix="> "
        )
    ]


def init():
    os.system("cls") if os.name == "nt" else os.system("clear")
    global name
    while True:
        printf("Enter your name.\n")
        name = input("> ")
        if navigate.yn_prompt(
            char_prompt=False, question=f"Are you sure your name is {name}?"
        ):
            break
        else:
            continue
    printf("Off we go!")
    print()
    time.sleep(1)


def room_1():
    global room_order
    print()
    printf(
        f"{name} {'enters' if room_order <= 0 else 'continues into'} the first room of the tunnel."
    )
    printf(f"What will {name} do?")
    room_order += 1
    choices = [
        "Look around",
        "Leave the room",
    ]
    while True:
        print()
        choice = get_choice(choices)
        if choice.lower() == "look around":
            global shield
            if not shield:
                printf(
                    f"{name} explores the room and discovers a shield that can be used 10 times."
                )
                shield = True
                print()
                choice1 = get_choice(["Stay", "Leave"])
                if choice1 == "Stay":
                    printf(f"{name} decides to remain in this room.")
                    continue
                elif choice1 == "Leave":
                    printf(
                        f"{name} opts not to interact further with this room and chooses to leave."
                    )
                    main()
                    break
            else:
                printf(f"{name} looks around but finds nothing of interest.")
        elif choice.lower() == "leave the room":
            printf(f"{name} decides to exit the room without further ado.")
            main()
            break


def room_2():
    global room_order
    print()
    printf(
        f"{name} {'enters' if room_order <= 0 else 'continues into'} the second room of the tunnel."
    )
    printf(f"What will {name} do?")
    room_order += 1
    choices = [
        "Look around",
        "Leave the room",
    ]
    while True:
        print()
        choice = get_choice(choices)
        if choice.lower() == "look around":
            global sword
            if not sword:
                printf(f"{name} explores the room and discovers a sword.")
                sword = True
                print()
                choice1 = get_choice(["Stay", "Leave"])
                if choice1 == "Stay":
                    printf(f"{name} decides to remain in this room.")
                    continue
                elif choice1 == "Leave":
                    printf(
                        f"{name} opts not to interact further with this room and chooses to leave."
                    )
                    main()
                    break
            else:
                printf(f"{name} looks around but finds nothing of interest.")
        elif choice.lower() == "leave the room":
            printf(f"{name} decides to exit the room without further ado.")
            main()
            break


def room_3():
    global room_order
    print()
    printf(
        f"{name} {'enters' if room_order <= 0 else 'continues through'} the third {'and final ' if room_order >= 2 else ''}chamber of the tunnel."
    )
    printf(
        f"At the end of the dimly lit corridor, {name} discovers 5 mysterious potions. Each vial glimmers with an otherworldly hue, promising unknown powers. The weight of their decision hangs heavy as they contemplate which potion to choose."
    )
    printf(
        f"Heart pounding, {name} steps into the expansive room beyond. The air thickens, suffused with ancient magic. Shadows dance across the walls, and there, coiled atop a massive treasure hoard, lies the dragon. Its scales shimmer like polished emeralds, its eyes bearing the weight of centuries."
    )
    printf(
        f"The dragon regards them with a mix of curiosity and disdain. It has grown weary of countless knights and heroes who dared challenge it, only to succumb to its fiery breath. But this time, something is different. {name} exudes determination and resolve."
    )
    printf(
        '"Finally," the dragon rumbles, its voice echoing through the chamber. "Someone competent and strong enough to face me."'
    )
    printf("The battle begins.")
    boss_fight()


def game_over():
    printf("GAME OVER", 0.25)
    time.sleep(0.5)
    if navigate.yn_prompt(char_prompt=False, question="Do you want to try again?"):
        global shield, sword, room_order
        shield = False
        sword = False
        room_order = 0
        main()
    else:
        input("Press enter to exit...\n")
        sys.exit(0)


def boss_fight():
    max_player_hp = 10
    max_dragon_hp = 100
    player_hp = max_player_hp
    dragon_hp = max_dragon_hp
    health_potions = 5
    shield_up = False
    shield_durability = 10

    dragon_attacks = [
        f"{name} felt the searing heat as the dragon made a magic blow, leaving them singed and disoriented.",
        f"'s massive wings swept through the air, slapping {name} away like a ragdoll, crashing onto the walls.",
        f"{name} watched in disbelief as the dragon attempted to use `/kill`, but somehow they managed to survive the deadly command.",
        f"The dragon's eyes glowed with intensity as it constructed a laser beam in its mouth. It fired at {name}, who{' bravely raised their shield and deflected the beam' if shield else ' barely managed to dodge, losing 3 HP in the process'}.",
        f"summoned a tempest of fire and engulfed {name} in scorching flames.",
        f"whispered an ancient curse, causing {name}'s limbs to turn to stone.",
        f"unleashed a sonic roar that shattered {name}'s eardrums and left them disoriented.",
        f"conjured a blizzard, freezing {name} in a solid ice block.",
        f"teleported behind {name} and delivered a swift tail swipe, sending them sprawling.",
        f"breathed toxic fumes, leaving {name} coughing and weakened.",
        f"used telekinesis to lift {name} off the ground and toss them like a ragdoll.",
        f"created an illusion of a bottomless pit, causing {name} to stumble and fall.",
        f"weaved a web of shadows, ensnaring {name} and draining their life force.",
        f"opened a portal to another dimension, briefly trapping {name} in a nightmarish realm.",
        f"chuckled amidst the chaos.\nIt had a vivid dream where it defeated {name} effortlessly.\nBut then it woke up, and reality hit.\nThe dragon lunged, cutting {name} into pieces with its razor-sharp claws.",
    ]

    turn = True

    while dragon_hp > 0 and player_hp > 0:
        os.system("cls") if os.name == "nt" else os.system("clear")

        player_hp = min(player_hp, max_player_hp)
        dragon_hp = min(dragon_hp, max_dragon_hp)

        print(f"{name}: {player_hp} / {max_player_hp}")
        print(f"Dragon: {dragon_hp} / {max_dragon_hp}")
        print()

        if turn:
            print(f"{name}'s turn")
            print()
            printf(f"What will {name} do?")
            print()
            choice = get_choice(["Attack", "Heal", "Block"])
            if choice == "Attack":
                if sword:
                    random_attack = probably(90)
                    printf(
                        f"{name} brandishes their newly acquired sword and attempts to strike the dragon{', inflicting a deep wound on the dragon.' if random_attack else ' but misses, hitting only air.'}"
                    )
                    if random_attack:
                        attack_damage = random.randint(5, 20)
                        printf(
                            f"The attack connects, and the dragon suffers (-{attack_damage} HP)"
                        )
                        dragon_hp -= attack_damage
                        turn = False
                    else:
                        printf(
                            f"The dragon seizes the opportunity, its powerful wings striking {name} with force."
                        )
                        if shield_up:
                            printf(
                                f"But unbeknownst to the dragon, {name} was prepared, shield raised, and the attack fails to harm them."
                            )
                            printf(
                                "The shield absorbs the impact, and its durability decreases."
                            )
                            shield_durability -= 1
                            printf(f"{shield_durability} uses remaining.")
                            shield_up = False
                        else:
                            printf(f"The attack took away 1 HP from {name}.")
                            player_hp -= 1
                        turn = True
                else:
                    printf(
                        f"Unfortunately, {name} lacks a sword, leaving them unable to take any action."
                    )
                    turn = False
            elif choice == "Heal":
                if health_potions <= 0:
                    printf(f"{name} ran out of health potions!")
                else:
                    printf(f"{name} used 1 potion. (+2 HP)")
                    health_potions -= 1
                    player_hp += 2
            elif choice == "Block":
                if shield:
                    printf(
                        f"{name} lifts up their shield, ready to block the dragon's attacks."
                    )
                    shield_up = True
                    turn = True
                else:
                    printf(f"{name} doesn't have a shield.")
            time.sleep(1)
        elif not turn:
            printf("The dragon's turn")
            print()
            seq = random.randint(0, len(dragon_attacks) - 1)

            if not dragon_attacks[seq].startswith(f"{name}"):
                printf(f"The dragon {dragon_attacks[seq]}")
            else:
                printf(dragon_attacks[seq])

            if not shield_up:
                if seq == 3:
                    printf(
                        f"The attack was so powerful that it took away 3 HP from {name}!"
                    )
                    player_hp -= 3
                else:
                    printf(f"The attack took away 1 HP from {name}.")
                    player_hp -= 1
            elif shield_up:
                printf(f"{name} used their shield.")
                if shield_durability <= 0:
                    printf("The shield broke.")
                else:
                    printf("They successfully blocked off the attacks.")
                    printf(
                        "The shield absorbs the impact, and its durability decreases."
                    )
                    shield_durability -= 1
                    printf(f"{shield_durability} times remaining.")
                    shield_up = False
            time.sleep(1)
            turn = True

    if player_hp <= 0:
        printf(f"{name} is now dead on the floor.")
        time.sleep(2)
        game_over()
    elif dragon_hp <= 0:
        printf(f"The dragon has now been defeated by {name}. The world lives on.")
        time.sleep(2)
        printf("Thanks for playing my game.", 0.075)
        input("Press enter to exit...")
        sys.exit(0)


def main():
    os.system("cls") if os.name == "nt" else os.system("clear")
    if room_order <= 0:
        printf(
            f"Welcome! Our dear adventurer {name} has been searching for a, well... an adventure, and they found a tunnel. A dark tunnel. {name} decides to approach the tunnel."
        )
    printf(
        f"{'Outside, t' if room_order > 0 else 'T'}here are 3 rooms.\nChoose a room that {name} would go to{' first' if room_order < 1 else ''}."
    )

    print()

    choice = get_choice(["First room", "Second room", "Third room"])
    if choice == "First room" or choice.lower() == "first":
        room_1()
    elif choice == "Second room" or choice.lower() == "second":
        room_2()
    elif choice == "Third room" or choice.lower() == "third":
        room_3()


if __name__ == "__main__":
    init()
    main()
