import requests
import logging
from loguru import logger
from decouple import config as env_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_activity_from_video(video_path):
    """
    Predicts the activity from a video by calling the activity classification API.
    Also determines if a flat floor assumption is valid based on the activity.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - str: The predicted activity (or None if prediction fails).
            - bool: True if a flat floor can be assumed, False otherwise.
    """
    predicted_activity = None
    flat_floor = False  # Default to False

    # List of activities where a flat floor can be assumed
    flat_floor_activities = [
        "squatting",
        "sit-to-stand",
        "walking",
        "running",
        "standing",
        "sitting",
    ]

    try:
        videollama_base = env_config("VIDEOLLAMA_URL", default="http://155.98.9.60:8400").rstrip("/")
        activity_api_url = f"{videollama_base}/predict"
        logger.info(
            f"Requesting activity prediction from {activity_api_url} for: {video_path}"
        )

        # Send video path as a form field — the VideoLLaMA container reads the
        # file directly from the shared volume (same path in both containers).
        data = {"video_path": video_path, "fps": "3", "max_frames": "20"}
        response = requests.post(activity_api_url, data=data, timeout=120)

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        activity_data = response.json()
        predicted_activity = activity_data.get("predicted_activity")
        logger.info(f"Predicted activity: {predicted_activity}")

        # Handle activities with slashes (e.g., "Squatting/Sit-to-stand")
        # Split by "/" and check if any part matches the allowed activities
        activity_parts = [
            part.strip().lower() for part in predicted_activity.split("/")
        ]
        matches_allowed_activity = any(
            part in flat_floor_activities for part in activity_parts
        )
        has_one_leg = "1 leg" in predicted_activity.lower()

        if matches_allowed_activity and not has_one_leg:
            flat_floor = True
            logger.info(
                f"Activity '{predicted_activity}' allows for flat floor assumption."
            )
        else:
            logger.info(
                f"Activity '{predicted_activity}' does not allow for flat floor assumption."
            )

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Connection to activity classifier at {activity_api_url} failed. Is the service running?"
        )
    except FileNotFoundError:
        logger.error(f"Video file not found: {video_path}")
    except requests.exceptions.HTTPError as e:
        # HTTPError is raised by response.raise_for_status() for 4xx/5xx status codes
        response = e.response
        if response is not None:
            status_code = response.status_code
            try:
                response_text = response.text
            except Exception:
                response_text = "Could not read response text"
            logger.error(
                f"HTTP error {status_code} from activity classifier at {activity_api_url}: {e}"
            )
            logger.error(f"Response text: {response_text}")
        else:
            logger.error(
                f"HTTP error from activity classifier at {activity_api_url}: {e}"
            )
    except requests.exceptions.RequestException as e:
        # For other request exceptions, try to get response details if available
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text
            logger.error(
                f"Request error {status_code} from activity classifier at {activity_api_url}: {e}"
            )
            logger.error(f"Response text: {response_text}")
        else:
            logger.error(f"Could not get activity prediction: {e}")

    return predicted_activity, flat_floor
