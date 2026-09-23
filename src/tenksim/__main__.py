from .cli import main

# ingest가 spawn 방식 프로세스를 띄우므로, 자식 프로세스가 main()을 다시 실행하지 않게 막는다
if __name__ == "__main__":
    main()
