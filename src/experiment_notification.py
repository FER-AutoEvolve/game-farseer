import dataclasses
import datetime
import enum
import logging
import threading
from typing import Any, Dict
from typing import Optional

import requests

from configuration import ExperimentNotificationConfiguration

'''
Module for sending experiment notifications to a configured HTTP endpoint.
Intended for external experiment-tracking components.
'''


@dataclasses.dataclass(frozen=True)
class ExperimentNotification:
    '''
    Data transfer object for experiment tracking notifications.
    '''
    component_name: str
    '''Name of the component emitting the event.'''
    event_type: "ExperimentEventTypes"
    '''Classification of the experiment event.'''
    message: str
    '''Human-readable event message.'''
    timestamp: datetime.datetime
    '''Timestamp of the notification.'''

    def to_json(self) -> Dict[str, Any]:
        '''
        Convert the ExperimentNotification to a JSON-serializable dictionary.
        '''
        payload: Dict[str, Any] = {
            "ComponentName": self.component_name,
            "EventType": self.event_type.value,
            "Message": self.message,
            "Timestamp": self.timestamp.isoformat()
        }
        return payload


class ExperimentEventTypes(enum.Enum):
    '''
    Enumeration of supported experiment event types.
    '''
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"
    INFO = "INFO"


@dataclasses.dataclass(frozen=True)
class ExperimentNotifier:
    '''
    Responsible for sending ExperimentNotification payloads to a configured HTTP endpoint.
    This notifier intentionally does not write local logs when sending experiment data.
    '''
    _config: ExperimentNotificationConfiguration

    def send(self, notification: ExperimentNotification, timeout: int = 3000) -> None:
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
# Define a new logging level that does not write to logs but triggers experiment notifications.

EXPERIMENT_LEVEL = 26
logging.addLevelName(EXPERIMENT_LEVEL, "EXPERIMENT")


def _logger_experiment(
    self,
    message: str,
    *args,
    event_type: ExperimentEventTypes = ExperimentEventTypes.INFO,
    **kwargs,
):
    '''
    Logger method that sends an experiment notification instead of writing to log handlers.
    It expects a globally configured notifier to be available as `global_experiment_notifier`.
    Parameters:
        message The message to include in the notification.
        event_type The event classification (default: ExperimentEventTypes.INFO).
        *args Additional positional arguments (ignored).
    '''
    global global_experiment_notifier

    ts = datetime.datetime.now(datetime.timezone.utc)

    if global_experiment_notifier is None:
        # Fallback: if notifier not configured, silently ignore to preserve behavior
        return

    notification = ExperimentNotification(
        component_name=global_experiment_notifier._config.component_name,
        event_type=event_type,
        message=message,
        timestamp=ts,
    )

    try:
        global_experiment_notifier.send(notification)
    except Exception:
        # Never raise from logging path
        return


# Attach method to logging.Logger
logging.Logger.experiment = _logger_experiment  # type: ignore[attr-defined]


global_experiment_notifier: Optional[ExperimentNotifier] = None
'''
Global experiment notifier instance. Must be configured via
`configure_experiment_notifier` before use.
'''


def configure_experiment_notifier(config: ExperimentNotificationConfiguration):
    '''
    Configure and initialize the global experiment notifier instance.
    Parameters:
        config Configuration for the experiment notifier.
    '''
    global global_experiment_notifier
    global_experiment_notifier = ExperimentNotifier(config)


__all__ = [
    "ExperimentEventTypes",
    "ExperimentNotification",
    "ExperimentNotificationConfiguration",
    "ExperimentNotifier",
    "configure_experiment_notifier",
]
