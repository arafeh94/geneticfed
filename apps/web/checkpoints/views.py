import logging

from debugpy.common.messaging import Request
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render

# Create your views here.
from src.apis import checkpoints_utils
from src.apis.utils import smooth


def index(request: HttpRequest):
    checkpoints: dict = checkpoints_utils.read()
    runs = list(checkpoints.keys())
    accuracies = []
    selected_runs = []
    if 'run' in request.POST:
        selected_runs = request.POST.getlist('run')
        if not isinstance(selected_runs, list):
            selected_runs = [selected_runs]
        for run_name in selected_runs:
            run_acc = [val['acc'] for round_id, val in checkpoints[run_name].history.items()]
            run_acc = smooth(run_acc)
            accuracies.append(run_acc)
    return render(request, 'index.html', {'runs': runs, 'acc': accuracies, 'selected': selected_runs})
