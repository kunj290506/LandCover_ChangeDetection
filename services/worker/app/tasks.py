import requests
from landcover_common.settings import Settings


def run_batch_job(payload: dict) -> dict:
    settings = Settings()
    results = []
    for pair in payload.get("pairs", []):
        response = requests.post(
            f"{settings.inference_url}/infer",
            json={
                "image_before_uri": pair["image_before_uri"],
                "image_after_uri": pair["image_after_uri"],
                "model_version": payload.get("model_version", "production"),
            },
            timeout=120,
        )
        response.raise_for_status()
        results.append(response.json())
    return {"outputs": results}
