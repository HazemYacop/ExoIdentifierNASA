from django.shortcuts import render


def home(request):
    """Render the landing page for the site."""
    return render(request, "main/home.html", {"page_title": "Exoplanet Explorer"})


def model(request):
    """Render the model exploration workspace."""
    return render(request, "main/model.html", {"page_title": "Model Workspace"})
