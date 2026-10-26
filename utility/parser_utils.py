# utility/parser_utils.py

def print_args(args, title="RUN CONFIGURATION"):
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("=" * 80)

    for k, v in sorted(vars(args).items()):
        print(f"{k:<25}: {v}")

    print("=" * 80 + "\n")


