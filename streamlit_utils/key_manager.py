class KeyManager:
    def __init__(self, prefix="widget"):
        self.prefix = prefix
        self.counter = 0

    def generate_key(self):
        """Element ID 생성

        Returns:
            str: "widget_{숫자}" 형식 Element ID
        """
        self.counter += 1
        return f"{self.prefix}_{self.counter}"


# ElementId 관리용 매니저 객체 생성
key_manager = KeyManager()
