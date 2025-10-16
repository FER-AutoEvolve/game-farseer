import dataclasses

from core import Result

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

@dataclasses.dataclass(frozen=True)
class KeypointNotificationConfiguration:
    ''' Configuration for the keypoint notification system.'''
    enabled: bool
    endpoint: str

    @staticmethod
    def from_dict(config: dict) -> Result['KeypointNotificationConfiguration']:
        ''' 
        Creates a KeypointNotificationConfiguration object from a dictionary.
        Args:
            config (dict): Dictionary containing configuration data.
        Returns:
            Result[KeypointNotificationConfiguration]: Result containing the KeypointNotificationConfiguration object or an error
        '''
        try:
            enabled = config.get("Enabled", True)
            endpoint = config.get("Endpoint", "")
            if not endpoint:
                return Result.err("KeypointNotification configuration requires 'Endpoint' to be set.")
            return Result.ok(KeypointNotificationConfiguration(
                enabled=enabled,
                endpoint=endpoint
            ))
        except ValueError as e:
            return Result.err(f"Invalid keypoint notification configuration: {e}")