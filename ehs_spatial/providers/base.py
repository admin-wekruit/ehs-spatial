class ProviderError(RuntimeError):
    def __init__(self, provider: str, operation: str, original_message: str) -> None:
        self.provider = provider
        self.operation = operation
        self.original_message = original_message
        super().__init__(f"{provider} {operation} failed: {original_message}")
