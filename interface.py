from core import Bot


def show() -> None:
    print("=" * 60)
    print("КОНСОЛЬНИЙ ЧАТ-БОТ НА БАЗІ МОДЕЛІ QWEN3-1.7B")
    print("Введіть ваше запитання та натисніть Enter.")
    print("Для завершення роботи введіть 'вихід' або 'exit'.")
    print("=" * 60)
    print("Завантаження моделі, будь ласка, зачекайте...\n")

    bot = Bot()
    print(f"Модель готова до роботи (пристрій: {bot.device}).\n")

    while True:
        try:
            user_input = input("Ви: ").strip()

            match user_input.lower():
                case "":
                    continue
                case "вихід" | "exit":
                    print("Бот: Роботу завершено. До зустрічі!")
                    break
                case _:
                    print("Бот генерує відповідь...", end="\r")
                    response = bot.generate_response(user_input)
                    print(" " * 30, end="\r")
                    print(f"Бот: {response}\n")

        except KeyboardInterrupt:
            print("\nБот: Програму перервано користувачем.")
            break
        except Exception as error:
            print(f"\nПомилка під час роботи ({type(error).__name__}): {error}\n")


if __name__ == "__main__":
    show()