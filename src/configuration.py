import dataclasses

__DEFAULT_FASTAPI_PORT__ = 8000
__DEFAULT_FASTAPI_HOST__ = "0.0.0.0"

@dataclasses.dataclass(frozen=True)
class FastApiConfiguration:
    '''
    Configuration for the Snake Auto-Designer FastAPI server.
    '''
    port: int = __DEFAULT_FASTAPI_PORT__
    host: str = __DEFAULT_FASTAPI_HOST__

    @staticmethod
    def from_dict(settings: dict) -> 'FastApiConfiguration':
        '''
        Creates a FastApiConfiguration from a settings dictionary.

        Args:
            settings (dict): Must contain optional keys "Port" and "Host".

        Returns:
            FastApiConfiguration: A new configuration object with defaults filled in.
        '''
        return FastApiConfiguration(
            port=settings.get("Port", __DEFAULT_FASTAPI_PORT__),
            host=settings.get("Host", __DEFAULT_FASTAPI_HOST__)
        )