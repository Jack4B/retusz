import sys

def run():
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from .cli import main
        sys.exit(main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "transfer":
        from .cli import transfer_main
        sys.exit(transfer_main(sys.argv[2:]))
    from .gui.app import main
    main()

if __name__ == "__main__":
    run()
