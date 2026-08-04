"""Re-export main for convenience."""

def main():
    from app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
