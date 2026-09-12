"""Execution selection is explicit; PostgreSQL remains the durable job outbox."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading
from typing import Callable, Protocol


class JobExecutor(Protocol):
    def submit(self, job_id: str) -> str: ...
    def status(self, executor_ref: str) -> dict: ...
    def cancel(self, executor_ref: str) -> bool: ...


class LocalJobExecutor:
    def __init__(self, worker: Callable[[str], object] | None = None, *, max_workers=2, python_executable=None, working_directory=None):
        self.worker = worker
        self.python_executable = python_executable or sys.executable
        self.working_directory = working_directory
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="panoptes-worker")
        self.futures, self.processes = {}, {}
        self.lock = threading.Lock()

    def _run(self, job_id, executor_ref):
        if self.worker:
            self.worker(job_id)
            return 0
        with self.lock:
            process = subprocess.Popen([self.python_executable, "-m", "panoptes_worker", "--job-id", job_id],
                cwd=self.working_directory, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes[executor_ref] = process
        return process.wait()

    def submit(self, job_id):
        reference = "local:" + job_id
        with self.lock:
            previous = self.futures.get(reference)
            if previous and previous.done() and (previous.cancelled() or previous.exception() is not None or previous.result() != 0):
                self.futures.pop(reference)
                self.processes.pop(reference, None)
            if reference not in self.futures:
                self.futures[reference] = self.pool.submit(self._run, job_id, reference)
        return reference

    def status(self, executor_ref):
        future = self.futures.get(executor_ref)
        if future is None:
            return {"status": "outcome_unknown"}
        if future.cancelled():
            return {"status": "cancelled"}
        if not future.done():
            return {"status": "running" if future.running() else "queued"}
        try:
            code = future.result()
            return {"status": "succeeded" if code == 0 else "failed", "exitCode": code}
        except Exception:
            return {"status": "failed"}

    def cancel(self, executor_ref):
        with self.lock:
            future = self.futures.get(executor_ref)
            if future is None:
                return False
            if future.cancel():
                return True
            process = self.processes.get(executor_ref)
            if process and process.poll() is None:
                process.terminate()
                return True
            return False

    def close(self):
        self.pool.shutdown(wait=True)


class ModalJobExecutor:
    def __init__(self, app_name, function_name, *, function=None):
        if function is None:
            if not app_name or not function_name:
                raise ValueError("Modal app and function are required")
            import modal
            function = modal.Function.from_name(app_name, function_name)
        self.function = function

    def submit(self, job_id):
        return self.function.spawn(job_id).object_id

    def status(self, executor_ref):
        import modal
        call = modal.FunctionCall.from_id(executor_ref)
        try:
            call.get(timeout=0)
            return {"status": "succeeded"}
        except TimeoutError:
            return {"status": "running"}
        except Exception:
            return {"status": "failed"}

    def cancel(self, executor_ref):
        import modal
        modal.FunctionCall.from_id(executor_ref).cancel()
        return True


def dispatch_pending(repository, executor):
    dispatched = []
    for job in repository.pending_jobs():
        # A lost dispatcher response can redeliver; claim_job fences execution.
        try:
            receipt = executor.submit(job["id"])
        except Exception:
            continue
        repository.mark_dispatched(job["id"], receipt)
        dispatched.append(job["id"])
    return dispatched
