from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .services import predict_proba


def home(request):
    """Render the landing page for the site."""
    return render(request, "main/home.html", {"page_title": "Exoplanet Explorer"})


def model(request):
    """Render the model exploration workspace."""
    get_token(request)
    return render(request, "main/model.html", {"page_title": "Model Workspace"})


@require_POST
def analyze_image(request):
    """Run the uploaded image through the classifier and report the result."""
    upload = request.FILES.get("image")
    if not upload:
        return JsonResponse({"error": "No image uploaded."}, status=400)

    try:
        probability, label = predict_proba(upload.read())
    except Exception as exc:  # pragma: no cover - serialized for client feedback
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse(
        {
            "label": label,
            "probability": probability,
            "confidence_percent": round(probability * 100, 2),
            "threshold": 0.5,
        }
    )
