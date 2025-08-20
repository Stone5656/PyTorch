import json
import tempfile
from django.http import JsonResponse, HttpRequest
from django.conf import settings

from mlmini.regression.infer import load_regression_predictor
from mlmini.classification.infer import load_classification_predictor

_regressor = None
_classifier = None

def _get_regressor():
    global _regressor
    if _regressor is None:
        model_root = getattr(settings, "MODEL_ROOT", None)
        if not model_root:
            return None
        _regressor = load_regression_predictor(model_root)
    return _regressor

def _get_classifier():
    global _classifier
    if _classifier is None:
        model_root = getattr(settings, "MODEL_ROOT", None)
        if not model_root:
            return None
        _classifier = load_classification_predictor(model_root)
    return _classifier

def regress_predict(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    regressor = _get_regressor()
    if regressor is None:
        return JsonResponse({"error": "MODEL_ROOT 未設定またはモデル未配置"}, status=500)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "JSON parse error"}, status=400)
    features = payload.get("features", payload)
    try:
        y = float(regressor(features))
        return JsonResponse({"y": y})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

def classify_predict(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    classifier = _get_classifier()
    if classifier is None:
        return JsonResponse({"error": "MODEL_ROOT 未設定またはモデル未配置"}, status=500)
    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"error": "image フィールドが必要です"}, status=400)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        for chunk in image.chunks():
            tmp.write(chunk)
        tmp.flush()
        try:
            label, score = classifier(tmp.name)
            return JsonResponse({"label": label, "score": float(score)})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
