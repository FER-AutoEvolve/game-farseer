import dataclasses
import enum
import datetime
import logging
from typing import Optional, Any, Dict
import threading
import requests
from configuration import KeypointNotificationConfiguration

'''
Module for sending keypoint notifications to a configured HTTP endpoint. Expectedly a web-game-wrapper component.
'''

class EventTypes(enum.Enum):
	'''
	Enumeration of event types in the entire evolutionary system.
	'''
	SUCCESS = "Success"
	'''Indicates a successful event'''
	FAILURE = "Failure"
	'''Indicates a failed event'''
	WARNING = "Warning"
	'''Indicates a warning event'''
	INFO = "Info"
	'''Indicates an informational event'''

@dataclasses.dataclass(frozen=True)
class EventNotification:
    '''
    Data transfer object for notifying about an event in the entire evolutionary system.
    '''
    source: str
    '''Source of the event (e.g., Game Farseer)'''
    event_type: EventTypes
    '''Type of the event'''
    details: str
    '''Details about the event'''
    timestamp: datetime.datetime
    '''Timestamp of the event'''
	
    def to_json(self) -> Dict[str, Any]:
        '''
        Convert the EventNotification to a JSON-serializable dictionary.
        '''
        payload = {
            "Source": self.source,
            "EventType": self.event_type.value,
            "Details": self.details,
            "Timestamp": self.timestamp.isoformat()
        }
        return payload

@dataclasses.dataclass(frozen=True)
class KeypointNotifier:
	'''
	Responsible for sending EventNotification payloads to a configured HTTP endpoint.
	This notifier intentionally does not write local logs when sending keypoints.
	'''
	_config: KeypointNotificationConfiguration

	def send(self, notification: EventNotification, timeout: int = 3000) -> None:
		'''
		Fire-and-forget send: dispatch the POST in a background daemon thread and return immediately.
		Any errors during sending are silently swallowed to avoid affecting application flow.
		'''
		if not self._config.enabled:
			return None

		headers = {"Content-Type": "application/json"}

		payload = notification.to_json()

		def _post():
			try:
					requests.post(self._config.endpoint, json=payload, headers=headers, timeout=timeout)
			except Exception:
				# Never raise from logging path
				return

		t = threading.Thread(target=_post, daemon=True)
		t.start()
		return None


# --- Logging integration ---
# Define a new logging level that does not write to logs but triggers keypoint notifications.

KEYPOINT_LEVEL = 25
logging.addLevelName(KEYPOINT_LEVEL, "KEYPOINT")


def _logger_keypoint(self, message: str, *args, event_type: EventTypes = EventTypes.INFO, source: Optional[str] = None, **kwargs):
    '''
    Logger method that will send a keypoint notification instead of writing to log handlers.
    It expects a globally configured notifier to be available as `global_keypoint_notifier`.
	Parameters:
		message The message to include in the notification.
		event_type The type of event (default: EventTypes.INFO).
		source Optional source override (default: "Game Farseer").
		*args Additional positional arguments (ignored).
    '''
    # Build notification
    ts = datetime.datetime.now(datetime.timezone.utc)
    src = "Game Farseer"
    notification = EventNotification(source=src, event_type=event_type, details=message, timestamp=ts)

    global global_keypoint_notifier

    if global_keypoint_notifier is None:
    	# Fallback: if notifier not configured, silently ignore to preserve behavior
        return

    try:
        global_keypoint_notifier.send(notification)
    except Exception:
    	# Never raise from logging path
    	return


# Attach method to logging.Logger
logging.Logger.keypoint = _logger_keypoint  # type: ignore[attr-defined]



global_keypoint_notifier: Optional[KeypointNotifier] = None
'''
Global keypoint notifier instance. Must be configured via `configure_keypoint_notifier` before use.
'''

def configure_keypoint_notifier(config: KeypointNotificationConfiguration):
	'''
	Configure and initialize the global keypoint notifier instance.
	Parameters:
		config Configuration for the keypoint notifier.
	'''
	global global_keypoint_notifier
	global_keypoint_notifier = KeypointNotifier(config)


__all__ = [
	"EventTypes",
	"EventNotification",
	"KeypointNotificationConfiguration",
	"KeypointNotifier",
	"configure_keypoint_notifier",
]

