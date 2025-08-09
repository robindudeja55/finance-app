from django.shortcuts import render
from django.http import JsonResponse


def healthz(request):
    return JsonResponse({"status": "ok"})
