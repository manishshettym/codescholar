from celery import Celery

# celery config
CELERY_BROKER_URL = "redis://localhost:6379"
CELERY_RESULT_BACKEND = "redis://localhost:6379"


# initialize celery app
def get_celery_app_instance(app):
    print("Retrieving a celery app instance...")
    celery = Celery(app.import_name, backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
